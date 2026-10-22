"""Speculative decoding (MTP/EAGLE-style) for AdaLLM.

v1 design (lossless, snapshot-restore for delta-net):
  1. Target decode step produces h_t (hidden feeding lm_head) + sampled token y_t.
  2. Draft layer runs k steps from (h_t, y_t) -> proposals [p1..pk] (greedy).
  3. VERIFY: run proposals through the TARGET as a mini chunked-prefill
     continuation (multi-token path: KV append + delta-state evolution),
     collecting logits at all k positions.
  4. ACCEPT: longest prefix of proposals matching target greedy. Always emit
     at least 1 token (the target's own greedy at position 1 = normal decode
     result, so worst case == plain decode + draft cost).
  5. On truncation (j < k accepted): restore delta state snapshot (conv+mamba
     for all 48 layers, one slot) and truncate KV to accepted length.

Enabled per-process with ADALLM_MTP=1 (k = ADALLM_MTP_K, default 3).
"""
import os
import torch
from torch import nn

_MTP_ENV = os.getenv("ADALLM_MTP", "0").strip().lower()
MTP_ENABLED = _MTP_ENV not in ("0", "false", "off", "")
MTP_K = max(1, int(os.getenv("ADALLM_MTP_K", "3")))


def snapshot_delta_state(runner, slot: int):
    return [(m.mamba_state[slot].clone() if m.mamba_state is not None else None,
             m.conv_state[slot].clone() if m.conv_state is not None else None)
            for m in runner.delta_modules]


def restore_delta_state(runner, slot: int, snap):
    for m, (ms, cs) in zip(runner.delta_modules, snap):
        if ms is not None:
            m.mamba_state[slot].copy_(ms)
        if cs is not None:
            m.conv_state[slot].copy_(cs)


class MTPRunner:
    """Owns the draft model; called by ModelRunner.run() after each decode step."""

    def __init__(self, model_dir: str, runner, dtype=torch.bfloat16):
        from adallm.architectures.mtp_draft import MTPDraft, load_mtp
        hf = runner.config.hf_config
        tc = getattr(hf, "text_config", hf)
        max_pos = min(getattr(tc, "max_position_embeddings", 32768) or 32768,
                      runner.config.max_model_len + 64)
        self.draft = MTPDraft(
            hidden_size=tc.hidden_size,
            num_heads=tc.num_attention_heads,
            num_kv_heads=tc.num_key_value_heads,
            head_dim=tc.head_dim,
            intermediate_size=getattr(tc, "intermediate_size", 17408),
            max_position=max_pos,
            rope_theta=getattr(tc, "rope_theta", 10000.0),
        ).to(runner.model.lm_head.weight.device).to(dtype)
        load_mtp(self.draft, model_dir, runner.model.lm_head.weight.device, dtype)
        self.draft.eval()
        # draft KV cache per sequence: slot -> dict
        self.kv = {}
        self.runner = runner
        lm = runner.model
        self.embed = lm.model.language_model.embed_tokens.weight.detach()
        self.lm_head_w = lm.lm_head.weight.detach()
        self.last_h = None  # [B, H] hidden from last target step
        self.last_tokens = None  # [B] sampled tokens from last target step
        self.k = MTP_K

    def capture(self, h, tokens):
        """Called after target forward: h [B,H] final-norm hidden, tokens [B]."""
        self.last_h = h.detach()
        self.last_tokens = tokens

    def propose(self, seq_ids: list[int], positions: list[int]) -> dict[int, list[int]]:
        """Draft k tokens for each sequence. Returns seq_id -> [p1..pk].
        v1: single-sequence only; returns {} otherwise."""
        if len(seq_ids) != 1:
            return {}
        with torch.inference_mode():
            sid, pos = seq_ids[0], positions[0]
            row = 0  # v1: single sequence -> row 0 of last_h
            h = self.last_h[row:row + 1].to(self.draft.fc.weight.dtype)
            tok = int(self.last_tokens[row])
            kv = self.kv.get(sid)
            if kv is None or kv["len"] != pos:
                # resync: rebuild cache empty at this length (draft quality hit
                # only on resync; steady state keeps cache warm)
                maxL = self.runner.config.max_model_len + 8
                dev = h.device
                kv = {"k": torch.zeros(1, self.draft.num_kv_heads, maxL, self.draft.head_dim,
                                        dtype=h.dtype, device=dev),
                      "v": torch.zeros(1, self.draft.num_kv_heads, maxL, self.draft.head_dim,
                                        dtype=h.dtype, device=dev),
                      "len": 0}
                self.kv[sid] = kv
            props = []
            cur_h, cur_tok, p = h, tok, pos
            for _ in range(self.k):
                e = self.embed[cur_tok].unsqueeze(0).to(h.dtype)
                dh = self.draft(cur_h, e, torch.tensor([p], device=h.device), kv_cache=kv)
                logits = (dh @ self.lm_head_w.t())  # bf16: no 4.7GB fp32 copy
                nt = int(logits[0].argmax())
                props.append(nt)
                # next draft input: reuse dh as pseudo-hidden, embed(nt)
                cur_h = dh
                cur_tok = nt
                p += 1
            # rewind cache len: we advanced k rows speculatively; verification
            # will tell how many were right. Caller restores kv["len"] = pos + j.
            return {sid: props}

    def trim_cache(self, sid: int, new_len: int):
        kv = self.kv.get(sid)
        if kv is not None and kv["len"] > new_len:
            kv["len"] = new_len

    def drop(self, sid: int):
        self.kv.pop(sid, None)
        self._row.pop(sid, None)

    _row: dict = {}
