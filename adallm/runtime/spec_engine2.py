"""Clean-room speculative decoding engine (v2) for AdaLLM. See module docstring
in file header of previous revision for design notes.

Data flow per spec step (single sequence, mid-decode):
  state invariants mid-decode: num_prefilled_tokens == num_tokens - 1
  (everything except newest token has KV/delta state written).

  1. DRAFT k tokens autoregressively on draft's own KV cache.
  2. VERIFY: rewind y_last, re-feed [y_last, p1..pk] as continuation prefill
     (num_prefilled_tokens = n-1, cur_chunk = k+1). Delta state snapshot taken
     first. Hidden rows [k+1, H]; logits = hidden @ W_head; row j validates
     proposal j (row j's position precedes p_{j+1}), row k = bonus.
  3. ACCEPT longest prefix; restore sequence + delta state; append finals;
     set num_prefilled_tokens = num_tokens - 1 (correction's KV is stale but
     the NEXT verify re-feeds it, overwriting before anything reads it).
  4. Pairing for next draft: (hidden[accepted], correction).

Plain steps (prefill, finish, exception) run through the original engine with a
run_model wrapper capturing hidden/logits so the first post-prefill pairing is
exact.
"""
from __future__ import annotations

import os
import torch

from adallm.architectures.mtp_draft import MTPDraft, load_mtp
from adallm.runtime.sequence import SequenceStatus as _SEQ_ST
_SEQ_FINISHED = _SEQ_ST.FINISHED


def _dbg(msg: str):
    if os.getenv("ADALLM_MTP_DEBUG"):
        print(f"[mtp2] {msg}", flush=True)


_DELTA_POOL_ATTRS = ("mamba_state", "conv_state")


def _delta_modules(mr):
    """Yield (module_name, module) for every layer owning a delta-state pool."""
    for name, m in mr.model.named_modules():
        if any(isinstance(getattr(m, a, None), torch.Tensor) for a in _DELTA_POOL_ATTRS):
            yield name, m


def _snapshot_delta(mr, slot: int):
    """Clone mamba_state/conv_state rows for this slot across all delta layers."""
    if not slot:
        return None
    snap = {}
    for name, m in _delta_modules(mr):
        for attr in _DELTA_POOL_ATTRS:
            pool = getattr(m, attr, None)
            if pool is not None:
                snap[(name, attr)] = pool[slot].clone()
    return (slot, snap) if snap else None


def _restore_delta(mr, snap):
    if snap is None:
        return
    slot, tensors = snap
    by_name = {}
    for name, m in _delta_modules(mr):
        by_name[name] = m
    for (name, attr), t in tensors.items():
        m = by_name.get(name)
        if m is not None:
            getattr(m, attr)[slot].copy_(t)


class SpecEngine2:
    def __init__(self, llm):
        self.llm = llm
        self.sched = llm.scheduler
        self.mr = llm.model_runner
        self.k = int(os.getenv("ADALLM_MTP_K", "3"))
        self.stats = {"spec_steps": 0, "plain_steps": 0, "tokens": 0,
                      "draft_accepted": 0, "draft_total": 0, "fallbacks": 0,
                      "p1_hit": 0, "p1_total": 0}
        dev = torch.device("cuda:0")
        self.draft = MTPDraft(max_position=16384).to(dev).to(torch.bfloat16)
        load_mtp(self.draft, self.mr.config.model, dev, torch.bfloat16)
        self.draft.eval()
        lm = self.mr.model
        self.embed_w = lm.model.language_model.embed_tokens.weight.detach()
        w = lm.lm_head.weight.detach()
        if w.numel() == 0:
            raise RuntimeError("lm_head weight empty (Marlin lm_head on? disable "
                               "ADALLM_LMHEAD_MARLIN for MTP — draft needs raw W)")
        self.lm_head_w = w.to(torch.float32 if w.dtype == torch.bfloat16 else w.dtype)
        self._kv = {}
        self._pending = None      # (h[M,1..,H] bf16, tok int) from last target run
        self._snap = None
        # capture wrapper on plain runs
        self._orig_run_model = self.mr.run_model
        self.mr.run_model = self._capture_run_model
        self._capture_hidden = None
        _dbg(f"init ok, k={self.k}")

    # ---------- capture wrapper (plain steps) ----------

    def _capture_run_model(self, input_ids, positions, is_prefill):
        # eager path only (we run with ADALLM_NO_COMPILE / eager in MTP mode)
        h = self.mr.model(input_ids, positions)
        logits = self.mr.model.compute_logits(h)
        self._capture_hidden = h.detach()
        self._last_logits = logits.detach()
        return logits

    # ---------- install ----------

    def install(self):
        self.llm.step = self._step

    # ---------- public step ----------

    def _step(self, return_updates: bool = False):
        sched = self.sched
        # prefill / waiting / chunked -> plain
        if sched.waiting or sched.chunked:
            out = self._plain()
        else:
            running = [s for s in sched.running if not s.is_finished]
            if len(running) == 1 and self._pending is not None:
                try:
                    out = self._spec(running[0])
                except Exception as e:
                    _dbg(f"spec failed: {e!r}; restore+plain")
                    self.stats["fallbacks"] += 1
                    if self._snap:
                        try:
                            # Round6-B: correct arity + also restore delta state
                            # (was _restore_seq(snap0, snap1): TypeError was
                            # swallowed by this except, leaving seq corrupt
                            # after every spec-step exception)
                            self._restore_seq(running[0], self._snap[0])
                            _restore_delta(self.mr, self._snap[1])
                        except Exception:
                            pass
                        self._snap = None
                    self._pending = None
                    out = self._plain()
            else:
                out = self._plain()
        # after plain decode step, refresh pairing from captured hidden
        if self._capture_hidden is not None and out is not None and out[1] < 0:
            seqs = [s for s in sched.running if not s.is_finished]
            if seqs:
                s = seqs[0]
                self._pending = (self._capture_hidden[-1:].to(torch.bfloat16),
                                 s.last_token)
        # Round6-B (queue-server deadlock fix): _plain/_spec return 3-tuples
        # (outputs, num_tokens, updates). The previous `out + ([],)` returned
        # updates ALWAYS empty, which starved the queue engine loop
        # (_engine_loop fans tokens to per-request queues from updates) and
        # llm.generate_stream: requests hung forever on q.get().
        if return_updates:
            return out          # (outputs, num_tokens, updates)
        assert out is not None
        return out[0], out[1]

    # ---------- plain ----------

    def _plain(self):
        seqs, is_prefill = self.sched.schedule()
        token_ids = self.mr.run(seqs, is_prefill)
        # Round6-B: return postprocess's per-token updates ((seq_id, tok, fin)
        # tuples) so queue-mode consumers receive tokens (same contract as
        # LLMEngine.step).
        updates = self.sched.postprocess(seqs, token_ids)
        outputs = [(s.seq_id, s.completion_token_ids, s.finish_reason)
                   for s in seqs if s.is_finished]
        num_tokens = sum(len(s) for s in seqs) if is_prefill else -len(seqs)
        self.stats["plain_steps"] += 1
        return outputs, num_tokens, updates

    # ---------- snapshot / restore ----------

    def _snap_seq(self, seq):
        bm = self.sched.block_manager
        last_blk_state = None
        if seq.block_table:
            lb = bm.blocks[seq.block_table[-1]]
            last_blk_state = (lb.hash, list(lb.token_ids))
        return {
            "token_ids": list(seq.token_ids), "num_tokens": seq.num_tokens,
            "num_prefilled_tokens": seq.num_prefilled_tokens,
            "cur_chunk": seq.cur_chunk, "chunk_final": seq.chunk_final,
            "finish_reason": seq.finish_reason,
            "block_table": list(seq.block_table),
            "last_blk": last_blk_state,
        }

    def _restore_seq(self, seq, snap):
        seq.token_ids = snap["token_ids"]
        seq.num_tokens = snap["num_tokens"]
        seq.num_prefilled_tokens = snap["num_prefilled_tokens"]
        seq.cur_chunk = snap["cur_chunk"]
        seq.chunk_final = snap["chunk_final"]
        seq.last_token = seq.token_ids[-1]
        seq.finish_reason = snap["finish_reason"]
        bm = self.sched.block_manager
        while len(seq.block_table) > len(snap["block_table"]):
            blk = seq.block_table.pop()
            bm.blocks[blk].reset()
            if blk not in bm.free_block_ids:
                bm.free_block_ids.append(blk)
        if seq.block_table and snap.get("last_blk") is not None:
            last = bm.blocks[seq.block_table[-1]]
            h, toks = snap["last_blk"]
            if h != -1:
                # re-finalize exactly as it was
                if last.hash != h:
                    if last.hash in bm.hash_to_block_id:
                        del bm.hash_to_block_id[last.hash]
                    last.update(h, toks)
                    bm.hash_to_block_id[h] = last.block_id
            else:
                if last.hash != -1:
                    if last.hash in bm.hash_to_block_id:
                        del bm.hash_to_block_id[last.hash]
                    last.reset()

    def _reconcile_blocks(self, seq, next_len: int | None = None):
        """Align the last block's hash state with what may_append will need
        for the append that takes the seq to `next_len` (default: current
        length + 1). may_append requires: block OPEN if the new length % bs
        in 2..bs (partial fill continues) or == 0 (it finalizes itself);
        block FINALIZED only when new length % bs == 1 (a fresh block is
        allocated). The spec verify/replay/restore dance around boundaries can
        leave stale state; compute the needed state from arithmetic."""
        bm = self.sched.block_manager
        bs = bm.block_size
        if not seq.block_table:
            return
        nl = next_len if next_len is not None else seq.num_tokens + 1
        need_finalized = (nl % bs == 1)
        last = bm.blocks[seq.block_table[-1]]
        if need_finalized and last.hash == -1:
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = bm.blocks[seq.block_table[-2]].hash if len(seq.block_table) > 1 else -1
            h = bm.compute_hash(token_ids, prefix)
            last.update(h, token_ids)
            bm.hash_to_block_id[h] = last.block_id
        elif not need_finalized and last.hash != -1:
            if last.hash in bm.hash_to_block_id:
                del bm.hash_to_block_id[last.hash]
            last.reset()

    # ---------- spec step ----------

    def _spec(self, seq):
        sched, mr, bm = self.sched, self.mr, self.sched.block_manager
        k = self.k
        h_cur, tok_cur = self._pending
        # KV pool sanity: leave headroom
        if len(bm.free_block_ids) < k // bm.block_size + 2:
            return self._plain()

        # ---- 1. draft ----
        kv = self._kv_for(seq.seq_id, len(seq) + k + 8)
        if os.getenv("ADALLM_MTP_NOKV"):
            kv["k"].zero_(); kv["v"].zero_(); kv["len"] = 0
        pos = len(seq) - 1
        if kv["len"] < pos - 1 and kv["len"] > 0 and pos - kv["len"] > self.k + 1:
            # gap too large (post-fallback divergence): full reset
            kv["k"].zero_(); kv["v"].zero_(); kv["len"] = 0
        proposals = []
        pos_start = pos
        h, tok = h_cur, tok_cur
        kv_len_start = kv["len"]
        with torch.inference_mode():
            for _ in range(k):
                e = self.embed_w[tok].unsqueeze(0).to(torch.bfloat16)
                dh = self.draft(h, e, torch.tensor([pos], device=h.device), kv_cache=kv)
                tok = int((dh.to(self.lm_head_w.dtype) @ self.lm_head_w.t())[0].argmax())
                proposals.append(tok)
                h = dh
                pos += 1

        # ---- 2. verify (span always [y, p1..pk] = k+1, bounded) ----
        self._snap = (self._snap_seq(seq), _snapshot_delta(mr, seq.delta_slot))
        y_last = seq.token_ids[-1]
        n_before = seq.num_tokens
        seq.token_ids = seq.token_ids[:-1]
        seq.num_tokens -= 1
        self._reconcile_blocks(seq)   # first verify append starts from len n-1
        for t in [y_last] + proposals:
            seq.append_token(t)
            if not bm.can_append(seq):
                raise RuntimeError("verify: no free KV blocks")
            _lb = bm.blocks[seq.block_table[-1]]
            if os.getenv("ADALLM_MTP_DEBUG"):
                _dbg(f"verify-append len={seq.num_tokens} mod={seq.num_tokens % bm.block_size} "
                     f"nblk={len(seq.block_table)} last.hash={_lb.hash}")
            bm.may_append(seq)
        seq.num_prefilled_tokens = n_before - 1
        seq.cur_chunk = k + 1
        try:
            sched.running.remove(seq)
        except ValueError:
            pass
        sched.chunked.append(seq)
        seqs_now, is_prefill = sched.schedule()
        assert is_prefill and seqs_now and seqs_now[0] is seq and len(seqs_now) == 1
        input_ids, positions = mr.prepare_prefill(seqs_now)
        with torch.inference_mode():
            hidden = mr.model(input_ids, positions)          # [k+1, H]
            logits = hidden.to(self.lm_head_w.dtype) @ self.lm_head_w.t()
        t_greedy = logits.argmax(dim=-1).tolist()            # row i predicts span[i+1]

        # ---- 3. accept ----
        accepted = 0
        for j in range(k):
            if proposals[j] == t_greedy[j]:
                accepted += 1
            else:
                break
        correction = t_greedy[accepted]
        finals = proposals[:accepted] + [correction]
        # Round6-B: truncate at the FIRST EOS — the plain engine stops there
        # (postprocess); finals bypass postprocess, so without this a mid-list
        # accepted EOS would be followed by extra tokens.
        if not seq.ignore_eos:
            _eos = sched.eos
            _eos_set = _eos if isinstance(_eos, set) else {_eos}
            for _i, _t in enumerate(finals[:-1]):
                if _t in _eos_set:
                    finals = finals[: _i + 1]
                    break

        # restore to pre-verify state
        self._restore_seq(seq, self._snap[0])
        _restore_delta(mr, self._snap[1])
        self._snap = None

        self._reconcile_blocks(seq)
        # append finals to the sequence (correction last)
        eos_set = sched.eos if not seq.ignore_eos else set()
        if not isinstance(eos_set, set):
            eos_set = {eos_set}
        fin = False
        reason = None
        for t in finals:
            seq.append_token(t)
            if not bm.can_append(seq):
                raise RuntimeError("accept: no free KV blocks")
            _lb = bm.blocks[seq.block_table[-1]]
            if seq.num_tokens % bm.block_size == 1 and _lb.hash == -1:
                # DUMP-STATE: the exact crash condition — dump everything
                raise RuntimeError(
                    f"DUMP-STATE accept-append bad: len={seq.num_tokens} bs={bm.block_size} "
                    f"nblk={len(seq.block_table)} table={seq.block_table} "
                    f"hashes={[bm.blocks[b].hash for b in seq.block_table]} "
                    f"snap_tbl={self._snap[0]['block_table'] if self._snap else None} "
                    f"snap_last={self._snap[0].get('last_blk') if self._snap else None} "
                    f"n_before={n_before} finals={finals} y={y_last}")
            bm.may_append(seq)

        # Round6-B: EOS short-circuit — if the last final is EOS the seq is
        # done; replay/3c are pointless (no further tokens needed) and the
        # upstream code never checks EOS on finals (only on z in 3c), so
        # without this the seq would decode all the way to max_tokens.
        if not seq.ignore_eos and finals and finals[-1] in eos_set:
            seq.finish_reason = "stop"
            seq.status = _SEQ_FINISHED
            try:
                sched.running.remove(seq)
            except ValueError:
                pass
            sched.block_manager.deallocate(seq)
            if sched.delta_slots is not None and seq.delta_slot:
                sched.delta_slots.free(seq.delta_slot)
                seq.delta_slot = 0
            self._kv.pop(seq.seq_id, None)
            self.stats["spec_steps"] += 1
            self.stats["tokens"] += len(finals)
            self.stats["draft_total"] += k
            self.stats["draft_accepted"] += accepted
            outputs = [(seq.seq_id, seq.completion_token_ids, seq.finish_reason)]
            updates = [(seq.seq_id, t, i == len(finals) - 1)
                       for i, t in enumerate(finals)]
            return outputs, -len(finals), updates

        # ---- 3b. replay [y, accepted finals except correction] through the
        # target so the delta state contains exactly the accepted tokens in
        # order; correction stays UNprocessed (next verify feeds it).
        replay_tokens = [y_last] + finals[:-1]
        with torch.inference_mode():
            # pop the correction so schedule() recomputes the chunk as exactly
            # [y, f1..f_acc] (it derives span from len(seq) - num_prefilled)
            correction_tok = seq.token_ids.pop()
            seq.num_tokens -= 1
            seq.last_token = seq.token_ids[-1]
            seq.num_prefilled_tokens = n_before - 1
            seq.cur_chunk = len(replay_tokens)
            try:
                sched.running.remove(seq)   # avoid double membership (running+chunked)
            except ValueError:
                pass
            sched.chunked.append(seq)
            seqs_r, is_p = sched.schedule()
            assert is_p and seqs_r and seqs_r[0] is seq
            in_r, pos_r = mr.prepare_prefill(seqs_r)
            assert in_r.tolist() == replay_tokens, f"replay span {in_r.tolist()} != {replay_tokens}"
            h_replay = mr.model(in_r, pos_r)     # [a+1, H]; last row = hidden at f_acc pos
            # re-append the correction (unprocessed, as intended)
            self._reconcile_blocks(seq)
            seq.append_token(correction_tok)
            if not bm.can_append(seq):
                raise RuntimeError("replay-append: no free KV blocks")
            bm.may_append(seq)
        # after replay, everything up to (not incl) correction is processed
        seq.num_prefilled_tokens = seq.num_tokens - 1

        # ---- 3c. advance the correction through ONE plain decode step:
        # writes its KV + delta state in decode order, yields the decode-context
        # hidden the draft expects as pairing (measured: draft needs
        # decode-context h; replay-prefill h alone → 10% vs 53% acceptance),
        # and its sampled token z is a free bonus token. postprocess appends z,
        # checks EOS/max_tokens, finishes and FREES delta slot + blocks.
        # decode branch calls may_append at CURRENT len N (preparing the block
        # for the unprocessed correction): need_finalized = (N % bs == 1)
        self._reconcile_blocks(seq, next_len=seq.num_tokens)
        seqs_d, is_dec = sched.schedule()
        assert not is_dec and seqs_d and seqs_d[0] is seq
        tok_ids_d = mr.run(seqs_d, is_dec)
        # Round6-B: keep z's postprocess updates for the fan-out (EOS/max_tokens
        # on the bonus token are checked there as usual).
        updates_d = sched.postprocess(seqs_d, tok_ids_d)

        fin = seq.is_finished
        reason = seq.finish_reason
        # multi-token steps can JUMP OVER postprocess's `== max_tokens` check
        # (e.g. 59 -> 62); enforce >= here so the seq always terminates.
        if not fin and seq.num_completion_tokens >= seq.max_tokens:
            seq.finish_reason = "length"
            seq.status = _SEQ_FINISHED
            fin = True
            reason = "length"
            try:
                sched.running.remove(seq)
            except ValueError:
                pass
            # free resources exactly like postprocess does (it only cleans up
            # seqs IT finished; this one finished in the spec path)
            sched.block_manager.deallocate(seq)
            if sched.delta_slots is not None and seq.delta_slot:
                sched.delta_slots.free(seq.delta_slot)
                seq.delta_slot = 0
            self._kv.pop(seq.seq_id, None)
        outputs = [(seq.seq_id, seq.completion_token_ids, seq.finish_reason)] if fin else []

        # pairing: decode-context hidden of the correction's forward (captured
        # by the run_model wrapper during 3c) + the NEW last token z
        h_pair = self._capture_hidden[-1:].to(torch.bfloat16) if self._capture_hidden is not None else h_replay[-1:].to(torch.bfloat16)
        tok_pair = seq.token_ids[-1]
        if not fin:
            self._pending = (h_pair, tok_pair)
        if os.getenv("ADALLM_MTP_DEBUG"):
            _dbg(f"acc={accepted} finals={finals} y={y_last} props={proposals} "
                 f"seq_len={seq.num_tokens} corr_pos={seq.num_tokens-1}")
        # draft KV trim (slot-indexed!): slot j holds the entry written by the
        # j-th draft call ever (this seq), at rope position offset+j where
        # offset is fixed at cache creation. This step's chain wrote k entries
        # at slots kv_len_start..kv_len_start+k-1 (for y,p1..pk at consecutive
        # rope positions). Keep entries for TRUE tokens only: y (slot
        # kv_len_start) and accepted proposals (next `accepted` slots). Drop
        # the rest (rejected hypotheticals + stale).
        kv["len"] = min(kv["len"], kv_len_start + accepted + 1)

        self.stats["spec_steps"] += 1
        self.stats["tokens"] += len(finals)
        self.stats["draft_total"] += k
        self.stats["draft_accepted"] += accepted
        self.stats["p1_total"] += 1
        if accepted >= 1:
            self.stats["p1_hit"] += 1
        if self.stats["spec_steps"] % 10 == 1:
            _dbg(f"spec {self.stats['spec_steps']}: props={proposals} acc={accepted} "
                 f"rate={self.stats['draft_accepted']/max(1,self.stats['draft_total']):.2f} "
                 f"p1={self.stats['p1_hit']}/{self.stats['p1_total']}")
        # Round6-B: synthesize per-token updates for the finals (they bypass
        # postprocess), then z's real updates from the 3c decode. The finished
        # flag rides the LAST update so the queue loop emits "done" exactly
        # when the seq terminates.
        updates = [(seq.seq_id, t, False) for t in finals] + list(updates_d)
        if updates:
            _sid, _tok, _f = updates[-1]
            updates[-1] = (_sid, _tok, _f or fin)
        return outputs, -len(finals), updates

    # ---------- misc ----------

    def _kv_for(self, seq_id, max_len):
        kv = self._kv.get(seq_id)
        dev = self.draft.fc.weight.device
        if kv is None or kv["k"].shape[2] < max_len:
            old = kv
            kv = {"k": torch.zeros(1, 4, max_len, 256, dtype=torch.bfloat16, device=dev),
                  "v": torch.zeros(1, 4, max_len, 256, dtype=torch.bfloat16, device=dev),
                  "len": 0}
            if old is not None:
                n = min(old["len"], max_len)
                kv["k"][:, :, :n] = old["k"][:, :, :n]
                kv["v"][:, :, :n] = old["v"][:, :, :n]
                kv["len"] = n
            self._kv[seq_id] = kv
        return kv

    def _eos_set(self):
        try:
            t = self.llm.tokenizer
            ids = set(t.eos_token_ids) if getattr(t, "eos_token_ids", None) else set()
            ids.add(t.eos_token_id)
            return {i for i in ids if i is not None}
        except Exception:
            return set()
