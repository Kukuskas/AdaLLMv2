"""Speculative decoding step (v1, single sequence, greedy) — engine integration.

Flow per decode step (replaces plain decode when ADALLM_MTP=1 and exactly one
RUNNING sequence exists):
  1. Normal target decode step -> y_t, h_t.
  2. Draft proposes k tokens from (h_t, y_t).
  3. Tentatively append proposals to the sequence; run target in "verify" mode:
     process the k proposal tokens as a mini continuation (multi-token path,
     like chunked prefill) capturing logits at each position.
  4. Accept longest greedy-matching prefix. Truncation repair:
     - restore delta-state snapshot (conv + mamba, all 48 delta layers)
     - free KV blocks allocated beyond the accepted length
     - rewind seq.token_ids
  5. Emit accepted tokens + the corrected next token (>=1 token/step always,
     identical distribution to plain greedy decode — lossless).

The verify pass returns logits per position; target greedy at position j
validates proposal j+1... (proposal j feeds logits at position j).
"""
import os
import torch

from adallm.runtime.spec_decode import (
    MTP_ENABLED, MTP_K, MTPRunner, snapshot_delta_state, restore_delta_state,
)


def _greedy_tokens(logits: torch.Tensor) -> list[int]:
    return logits.argmax(dim=-1).tolist()


class SpecEngine:
    """Attached to LLMEngine when MTP_ENABLED; hooks step()."""

    def __init__(self, llm):
        self.llm = llm
        self.orig_step = llm.step  # BEFORE serve.py rebinds llm.step
        self.mr = llm.model_runner
        self.runner = MTPRunner(llm.model_runner.config.model, self.mr)
        self.k = self.runner.k
        self.snap = None
        self.finished_outputs = []
        self.stats = {"steps": 0, "tokens": 0, "draft_tokens_ok": 0}
        # hook to capture final hidden: patch compute_logits is invasive;
        # instead re-run lm_head on h captured from run_model's return path.
        # run_model returns compute_logits(model(...)). For eager we can call
        # model() again cheaply? No. Instead: monkeypatch mr.run_model to
        # capture the hidden before lm_head.
        self._orig_run_model = self.mr.run_model
        self.mr.run_model = self._run_model_capture
        self.last_logits = None

    # -- capture wrapper: remember last hidden (row-aligned with input batch)
    def _run_model_capture(self, input_ids, positions, is_prefill):
        if is_prefill or self.mr.enforce_eager or input_ids.size(0) > 512:
            h = self.mr.model(input_ids, positions)
            logits = self.mr.model.compute_logits(h)
            if getattr(self, "_verify_mode", False):
                self.verify_hidden = h.detach()
            self.last_h = h.detach()
            self.last_logits = logits.detach()
            self.last_h_is_prefill = is_prefill
            return logits
        # graph path: outputs var IS the hidden
        logits = self._orig_run_model(input_ids, positions, is_prefill)
        self.last_h = self.mr.graph_vars["outputs"][: input_ids.size(0)].detach()
        self.last_logits = logits.detach()
        self.last_h_is_prefill = False
        return logits

    def step(self) -> list[tuple[int, int, bool]]:
        """One spec-decode step. Returns update tuples like scheduler.postprocess."""
        sched = self.llm.scheduler
        if sched.is_finished():
            return []
        running = list(sched.running)
        import os as _os
        if _os.getenv("ADALLM_DEBUG_MTP"):
            print(f"[mtp-dbg] running={len(running)} chunked={len(sched.chunked)} nct={[r.num_completion_tokens for r in running]} finished={sched.is_finished()}", flush=True)
        # decode mode: single running seq, past prefill (has emitted tokens),
        # no in-flight prefill chunks waiting
        in_decode = (len(running) == 1 and not sched.chunked
                     and running[0].num_completion_tokens >= 1)
        if not in_decode:
            # fall back to plain engine step — harvest its outputs for
            # finished seqs so serve.py's wrapper can deliver them
            outputs, num_tokens, updates = self.orig_step(return_updates=True)
            for seq_id, token_ids, fin_reason in outputs:
                self.finished_outputs.append((seq_id, token_ids, fin_reason))
            return updates
        seq = running[0]
        mr = self.mr

        # ---- 1. normal decode step for the sequence ----
        seqs_now, is_prefill = sched.schedule()
        assert not is_prefill
        # capture logits of this decode step via the patched run_model
        input_ids_d, positions_d = mr.prepare_decode(seqs_now) if False else (None, None)
        tok_ids = mr.run(seqs_now, is_prefill)
        # greedy argmax of the decode-step logits (validates proposals[0]);
        # temperature=0 sampling == greedy for our use case
        dec_logits_argmax = None
        try:
            dl = self.last_logits
            if dl is not None:
                dec_logits_argmax = int(dl[-1].argmax())
        except Exception:
            dec_logits_argmax = None
        updates = sched.postprocess(seqs_now, tok_ids)
        if seq.is_finished:
            self.runner.drop(seq.seq_id)
            return updates
        y_t = seq.last_token
        self.last_verify_tok = y_t
        if getattr(self, "next_h", None) is not None:
            # previous step's verify left the correct (h, tok) pairing
            y_t = self.next_tok
            h_t = self.next_h
            self.next_h = None
        else:
            h_t = self.last_h[-1:]
        pos = len(seq) - 1  # position of y_t (0-based index of last token)

        # ---- 2. draft proposals ----
        self.runner.last_h = h_t  # [1, H]
        self.runner.last_tokens = [y_t]
        proposals = self.runner.propose([seq.seq_id], [len(seq) - 1]).get(seq.seq_id)
        if not proposals:
            return updates
        k = len(proposals)

        # ---- 3. verify: tentative append + multi-token continuation ----
        self.snap = snapshot_delta_state(mr, seq.delta_slot) if seq.delta_slot else None
        num_tokens_before = len(seq)  # INCLUDES y_t (appended by postprocess)
        blocks_before = list(seq.block_table)
        bm = sched.block_manager
        # verify re-feeds [y_t, p1..pk]: rewind y_t first
        seq.token_ids = seq.token_ids[:-1]
        seq.num_tokens -= 1
        seq.last_token = seq.token_ids[-1]
        # free the block y_t's write may have crossed into (approximation:
        # re-run may_append bookkeeping on re-append below)
        verify_tokens = [int(self.last_verify_tok)] + [int(p) for p in proposals]
        for p in verify_tokens:
            seq.append_token(p)
            if not bm.can_append(seq):
                self._rewind(seq, num_tokens_before, blocks_before)
                return updates
            bm.may_append(seq)
        # run the k proposal tokens as a continuation (prefill-style multi-token)
        seq.num_prefilled_tokens = 0  # force full-context path? NO: use cached path
        # prepare_prefill computes start = num_cached + num_prefilled; we want
        # start = num_tokens_before (tokens after that are our proposals).
        seq.num_prefilled_tokens = num_tokens_before
        seq.num_cached_tokens = 0
        seq.cur_chunk = k
        seq.chunk_final = True
        self._verify_mode = True
        try:
            input_ids, positions = mr.prepare_prefill([seq])
            # NOTE: run_model in prefill mode scores ONLY the last position
            # (lm_head picks last_indices). We need logits at ALL k positions:
            # call the model directly, then lm_head on every row.
            h_verify = mr.model(input_ids, positions)          # [k, H]
            # raw matmul (lm_head module slices last row under prefill context)
            w = mr.model.lm_head.weight
            logits = h_verify @ w.t()                            # [k, vocab]
            self.last_h = h_verify.detach()  # capture for next draft round
        except Exception:
            # restore and bail out with plain step result
            self._rewind(seq, num_tokens_before, blocks_before)
            return updates
        t_greedy = _greedy_tokens(logits)  # [k+1] rows; row j validates proposals[j]
        accepted = 0
        for j in range(k):
            if proposals[j] == t_greedy[j]:
                accepted += 1
            else:
                break
        # correction = target token at first mismatch (or bonus if all accepted)
        correction = t_greedy[accepted]
        final_tokens = proposals[:accepted] + [correction]
        # next draft pairing: h that PRODUCED the correction, token = correction
        vh = getattr(self, "verify_hidden", None)
        if vh is not None and vh.size(0) > accepted:
            self.next_h = vh[accepted:accepted+1].clone()
            self.next_tok = correction
        # clamp to max_tokens budget
        room = seq.max_tokens - seq.num_completion_tokens
        if room <= 0:
            return updates
        final_tokens = final_tokens[:room]
        # rewind to num_tokens_before then re-append final tokens
        self._rewind(seq, num_tokens_before, blocks_before)
        for t in final_tokens:
            seq.append_token(t)
        # rewrite the updates list: original update had y_t; now emit all final tokens
        fin = False
        eos_set = sched.eos if not isinstance(sched.eos, set) else sched.eos
        for t in final_tokens:
            hit_eos = (t in eos_set) if isinstance(eos_set, set) else (t == eos_set)
            fin = fin or hit_eos
        if fin or seq.num_completion_tokens >= seq.max_tokens:
            self.finished_outputs.append((seq.seq_id, list(seq.completion_token_ids), "stop" if fin else "length"))
            print(f"[mtp] seq {seq.seq_id} FINISHED: {len(seq.completion_token_ids)} tokens, stats={self.stats}", flush=True)
            seq.status = __import__("adallm.runtime.sequence", fromlist=["SequenceStatus"]).SequenceStatus.FINISHED
            seq.finish_reason = "stop" if fin else "length"
            sched.block_manager.deallocate(seq)
            if sched.delta_slots is not None and seq.delta_slot:
                sched.delta_slots.free(seq.delta_slot)
                seq.delta_slot = 0
            if seq in sched.running:
                sched.running.remove(seq)
            self.runner.drop(seq.seq_id)
        # per-token updates, same semantics as scheduler.postprocess
        # (generate() rebuilds the full list from seq.completion_token_ids;
        #  generate_stream yields each update's token individually)
        new_updates = [(seq.seq_id, t, False) for t in final_tokens]
        if fin or new_updates == []:
            new_updates[-1] = (seq.seq_id, new_updates[-1][1], True)
        # stats
        self.stats["steps"] += 1
        self.stats["tokens"] += len(final_tokens)
        self.stats["draft_tokens_ok"] += accepted
        import os as _os
        if _os.getenv("ADALLM_DEBUG_MTP"):
            print(f"[mtp] step: proposals={proposals} accepted={accepted} final={final_tokens} stats={self.stats}", flush=True)
        # restore draft cache length to after accepted tokens
        self.runner.trim_cache(seq.seq_id, len(seq) - 1)
        self.snap = None  # drop snapshot reference promptly
        return new_updates

    def _rewind(self, seq, num_tokens_before, blocks_before):
        # truncate token ids
        seq.token_ids = seq.token_ids[:num_tokens_before]
        seq.num_tokens = num_tokens_before
        seq.last_token = seq.token_ids[-1]
        # KV blocks: return any blocks appended during tentative verify to
        # the free list (they were allocated by may_append above)
        bm = self.llm.scheduler.block_manager
        while len(seq.block_table) > len(blocks_before):
            blk = seq.block_table.pop()
            if blk not in bm.free_block_ids:
                bm.free_block_ids.append(blk)
                bm.blocks.pop(blk, None)
        # delta state
        if self.snap is not None and seq.delta_slot:
            restore_delta_state(self.mr, seq.delta_slot, self.snap)
        self.snap = None
