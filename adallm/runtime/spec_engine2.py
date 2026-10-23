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
import time

import torch

from adallm.architectures.mtp_draft import MTPDraft, load_mtp
from adallm.infra.context import get_context
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
        # Round7-A A/B debug switch: force the old restore+replay path even on
        # full accept (measurement only; default off = fast path enabled).
        self._slowpath = bool(os.getenv("ADALLM_MTP_SLOWPATH"))
        self.stats = {"spec_steps": 0, "plain_steps": 0, "tokens": 0,
                      "draft_accepted": 0, "draft_total": 0, "fallbacks": 0,
                      "p1_hit": 0, "p1_total": 0, "full_accept": 0,
                      "t_draft": 0.0, "t_verify": 0.0, "t_restore": 0.0,
                      "t_replay": 0.0, "t_3c": 0.0}
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
                    if os.getenv("ADALLM_MTP_DEBUG"):
                        import traceback
                        traceback.print_exc()
                        s = running[0]
                        bm_ = self.sched.block_manager
                        print(f"[mtp2-fb] len={s.num_tokens} bs={bm_.block_size} "
                              f"tbl={len(s.block_table)} snap={'yes' if self._snap else 'none'} "
                              f"npf={s.num_prefilled_tokens} chunk={s.cur_chunk} "
                              f"in_chunked={s in self.sched.chunked} "
                              f"in_running={s in self.sched.running}", flush=True)
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
        # after plain decode/prefill step, refresh pairing from captured hidden
        if self._capture_hidden is not None and out is not None and out[1] != 0:
            # Round7-A: also refresh after a FINAL prefill chunk (out[1] > 0).
            # Previously only decode steps (out[1] < 0) refreshed, so on every
            # request AFTER the first, the first spec step ran with the
            # PREVIOUS request's stale (h, tok) pair — junk drafts and an
            # acceptance dip on step 1. capture_hidden[-1] is the hidden of the
            # last prompt token; postprocess has appended the first sampled
            # token, so (h, last_token) is the exact MTP pairing. Intermediate
            # chunks are excluded: their seq is still in `chunked`, not
            # `running`.
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

    # ---------- L1b: graphed verify/replay forwards ----------

    def _ensure_verify_graph(self, span: int):
        """Capture (once per span) the fixed-shape span-row continuation
        forward. Requires non-eager engine + ADALLM_PAGED_VERIFY (the paged
        splitk path is device-indexed, capture-safe; the gather path has
        .item() syncs and cannot be captured)."""
        try:
            ok = self.mr.capture_verify_graph(span_rows=span)
        except Exception as e:
            _dbg(f"verify-graph capture failed: {e!r}")
            ok = False
        return self.mr.verify_graph_vars if ok else None

    def _span_forward(self, input_ids: torch.Tensor, positions: torch.Tensor):
        """Fixed-shape span-row prefill-continuation forward: per-span graph
        replay if captured, else eager. Returns hidden [span, H]."""
        bs = input_ids.shape[0]
        entry = getattr(self.mr, "verify_graphs", {}).get(bs)
        if entry is not None and os.getenv("ADALLM_MTP_GRAPH", "0") == "1":
            graph, gv = entry
            ctx = get_context()
            gv["input_ids"].copy_(input_ids)
            gv["positions"].copy_(positions)
            gv["slot_mapping"][:] = ctx.slot_mapping
            gv["delta_slot"][:] = ctx.delta_slots
            bt = ctx.block_tables
            gv["block_tables"].zero_()
            gv["block_tables"][0, : bt.shape[1]].copy_(bt[0])
            gv["cu_k"][1] = ctx.cu_seqlens_k[1]
            graph.replay()
            return gv["hidden"]
        hidden = self.mr.model(input_ids, positions)
        return hidden

    def _verify_forward(self, input_ids: torch.Tensor, positions: torch.Tensor):
        """Backward-compat alias: 4-row verify via _span_forward."""
        return self._span_forward(input_ids, positions)

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
        _t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(k):
                e = self.embed_w[tok].unsqueeze(0).to(torch.bfloat16)
                dh = self.draft(h, e, torch.tensor([pos], device=h.device), kv_cache=kv)
                tok = int((dh.to(self.lm_head_w.dtype) @ self.lm_head_w.t())[0].argmax())
                proposals.append(tok)
                h = dh
                pos += 1
        torch.cuda.synchronize(); self.stats["t_draft"] += time.perf_counter() - _t0

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
        _t0 = time.perf_counter()
        with torch.inference_mode():
            # Round8-C L1b: graphed verify (fixed span k+1) if enabled+captured
            if os.getenv("ADALLM_MTP_GRAPH", "0") == "1" and \
                    getattr(mr, "verify_graph", None) is None:
                self._ensure_verify_graph(k + 1)
            hidden = self._verify_forward(input_ids, positions)     # [k+1, H]
            logits = hidden.to(self.lm_head_w.dtype) @ self.lm_head_w.t()
        torch.cuda.synchronize(); self.stats["t_verify"] += time.perf_counter() - _t0
        t_greedy = logits.argmax(dim=-1).tolist()            # row i predicts span[i+1]
        if os.getenv("ADALLM_MTP_DEBUG_TIE"):
            # Round7-A diagnostic: verify logits are fp32 (hidden @ W fp32)
            # while plain decode uses bf16 F.linear — near-ties can argmax
            # differently. Count flips + top-2 margins when enabled.
            lg16 = (hidden.to(torch.bfloat16) @ self.mr.model.lm_head.weight.detach().t()
                    if hasattr(self.mr.model, "lm_head") else None)
            if lg16 is not None:
                t16 = lg16.float().argmax(dim=-1).tolist()
                top2 = logits.float().topk(2, dim=-1)
                for i, (a, b) in enumerate(zip(t_greedy, t16)):
                    if a != b:
                        m = (top2.values[i, 0] - top2.values[i, 1]).item()
                        print(f"[mtp2-tie] row{i}: fp32={a} bf16={b} top2_margin={m:.4f}",
                              flush=True)

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

        # Round7-A: exact max_tokens enforcement — a multi-token step can jump
        # OVER the limit (e.g. 79 -> 83 with the old `>=`-only check leaving
        # 81-83 tokens emitted). Truncate finals so the step lands on it
        # exactly; if the limit is reached before the correction, 3c (and its
        # bonus z) is skipped via the length short-circuit below.
        # NOTE: seq currently holds the VERIFY span [y, p1..pk] — k proposals
        # beyond n_before — so subtract k to get the true emitted-so-far count.
        _room = seq.max_tokens - (seq.num_completion_tokens - k)
        if _room > 0 and len(finals) > _room:
            finals = finals[:_room]

        # Round7-A: on FULL accept (accepted == k) the verify forward already
        # processed [y, p1..pk] and left delta+KV state correct through p_k.
        # The slow path's restore+replay would re-feed the exact same tokens
        # at the same positions from the same (restored) state — bit-identical
        # writes — so they are pure waste (~290ms of the ~320ms eager step).
        # Only the correction is unprocessed: append it WITHOUT a forward and
        # jump straight to the 3c decode step (which writes its KV+delta in
        # decode order and provides the decode-context pairing hidden).
        full = accepted == k and len(finals) == k + 1 and not self._slowpath
        h_replay = None
        # append finals to the sequence (correction last)
        eos_set = sched.eos if not seq.ignore_eos else set()
        if not isinstance(eos_set, set):
            eos_set = {eos_set}
        fin = False
        reason = None
        if full:
            # discard the snapshot WITHOUT restoring: verify's appends are the
            # truth now. Sequence already contains [.., y, p1..pk].
            self._snap = None
            _t0 = time.perf_counter()
            self._reconcile_blocks(seq, next_len=seq.num_tokens + 1)
            seq.append_token(correction)
            if not bm.can_append(seq):
                raise RuntimeError("fastpath correction append: no free KV blocks")
            bm.may_append(seq)
            # invariant: everything except the newest token processed —
            # correction's KV row is stale garbage, but nothing reads it
            # before the 3c decode below overwrites it (same slot/position).
            seq.num_prefilled_tokens = seq.num_tokens - 1
            torch.cuda.synchronize(); self.stats["t_restore"] += time.perf_counter() - _t0
            self.stats["full_accept"] += 1
        else:
            # restore to pre-verify state
            _t0 = time.perf_counter()
            self._restore_seq(seq, self._snap[0])
            _restore_delta(mr, self._snap[1])
            self._snap = None

            # Round7-A boundary fix (pre-existing bug): a restore at
            # len % bs == 1 is LAGGED — the block for the unprocessed tail
            # token is not yet in the table (plain protocol allocates it in
            # the decode that processes that token). The accept-append loop's
            # may_append only allocates when an append lands at %bs == 1;
            # starting from %bs == 1 the first append lands at %bs == 2, so
            # positions past the boundary never get a block and replay's
            # prepare_prefill IndexErrors (observed: 72/73 spec steps fell
            # back permanently after the first boundary crossing). Finalize
            # the full last block and pre-allocate the tail block.
            if (seq.num_tokens % bm.block_size == 1
                    and len(seq.block_table) == seq.num_blocks - 1):
                self._reconcile_blocks(seq, next_len=seq.num_tokens)  # finalize full last
                nb = bm._allocate_block(bm.free_block_ids[0])
                nb.precision = bm._precision_for_block_index(
                    len(seq.block_table), len(seq.block_table) + 1)
                seq.block_table.append(nb.block_id)
            else:
                self._reconcile_blocks(seq)
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
            torch.cuda.synchronize(); self.stats["t_restore"] += time.perf_counter() - _t0

        # Round6-B: EOS short-circuit — if the last final is EOS the seq is
        # done; replay/3c are pointless (no further tokens needed) and the
        # upstream code never checks EOS on finals (only on z in 3c), so
        # without this the seq would decode all the way to max_tokens.
        # Round7-A: also short-circuit when the (possibly truncated) finals
        # land exactly on max_tokens — running 3c anyway would emit bonus
        # token z and overshoot the limit (81 of 80).
        _hit_len = seq.num_completion_tokens >= seq.max_tokens
        if ((not seq.ignore_eos and finals and finals[-1] in eos_set) or _hit_len):
            seq.finish_reason = "stop" if (finals and finals[-1] in eos_set
                                           and not seq.ignore_eos) else "length"
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
        # Round7-A: SKIPPED on full accept — verify already produced those
        # exact writes; only the correction (appended above) is unprocessed.
        if not full:
            _t0 = time.perf_counter()
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
                # Round8-C L1b: graphed replay (fixed span a+1) when enabled
                if os.getenv("ADALLM_MTP_GRAPH", "0") == "1" and in_r.shape[0] not in getattr(mr, "verify_graphs", {}):
                    self._ensure_verify_graph(in_r.shape[0])
                h_replay = self._span_forward(in_r, pos_r)     # [a+1, H]; last row = hidden at f_acc pos
                # re-append the correction (unprocessed, as intended)
                self._reconcile_blocks(seq)
                seq.append_token(correction_tok)
                if not bm.can_append(seq):
                    raise RuntimeError("replay-append: no free KV blocks")
                bm.may_append(seq)
            # after replay, everything up to (not incl) correction is processed
            seq.num_prefilled_tokens = seq.num_tokens - 1
            torch.cuda.synchronize(); self.stats["t_replay"] += time.perf_counter() - _t0

        # ---- 3c. advance the correction through ONE plain decode step:
        # writes its KV + delta state in decode order, yields the decode-context
        # hidden the draft expects as pairing (measured: draft needs
        # decode-context h; replay-prefill h alone → 10% vs 53% acceptance),
        # and its sampled token z is a free bonus token. postprocess appends z,
        # checks EOS/max_tokens, finishes and FREES delta slot + blocks.
        # decode branch calls may_append at CURRENT len N (preparing the block
        # for the unprocessed correction): need_finalized = (N % bs == 1)
        # Round7-A wrong-slot fix: at N % bs == 1 the decode-append protocol
        # expects the table LAGGED (len num_blocks-1, last block finalized) —
        # schedule's may_append then allocates the final block and
        # prepare_decode maps the slot correctly. Our explicit append above
        # left it CURRENT (extra open block allocated): undo that allocation,
        # else may_append allocates one block TOO MANY and the correction's
        # KV silently lands in the wrong block (position N-1 read as garbage).
        if seq.num_tokens % bm.block_size == 1 and len(seq.block_table) == seq.num_blocks:
            blk = seq.block_table.pop()
            _b = bm.blocks[blk]
            if _b.hash != -1:
                bm.hash_to_block_id.pop(_b.hash, None)
            _b.reset()
            if blk not in bm.free_block_ids:
                bm.free_block_ids.append(blk)
        # invariant: entering the decode schedule, table covers exactly the
        # blocks may_append will need (lagged by one iff N % bs == 1)
        assert len(seq.block_table) == seq.num_blocks - (seq.num_tokens % bm.block_size == 1)
        self._reconcile_blocks(seq, next_len=seq.num_tokens)
        _t0 = time.perf_counter()
        seqs_d, is_dec = sched.schedule()
        assert not is_dec and seqs_d and seqs_d[0] is seq
        tok_ids_d = mr.run(seqs_d, is_dec)
        # Round6-B: keep z's postprocess updates for the fan-out (EOS/max_tokens
        # on the bonus token are checked there as usual).
        updates_d = sched.postprocess(seqs_d, tok_ids_d)
        torch.cuda.synchronize(); self.stats["t_3c"] += time.perf_counter() - _t0

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
        if self._capture_hidden is not None:
            h_pair = self._capture_hidden[-1:].to(torch.bfloat16)
        elif h_replay is not None:
            h_pair = h_replay[-1:].to(torch.bfloat16)
        else:
            raise RuntimeError("no pairing hidden: 3c capture missing on fast path")
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
