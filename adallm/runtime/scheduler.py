import os
from collections import deque

from adallm.settings import Config
from adallm.runtime.sequence import Sequence, SequenceStatus
from adallm.runtime.block_manager import BlockManager
from adallm.runtime.delta_slots import DeltaSlotManager


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        # ---- Round3-B: decode/prefill interleave (env-gated, default OFF) ----
        # ADALLM_INTERLEAVE=N (N>0): when decode seqs are RUNNING and prefill
        # work is pending, run at most one prefill step every N scheduler
        # steps (N-1 decode steps between prefill chunks). N=0 => legacy
        # prefill-priority (decode starves for a whole chunk duration).
        self.interleave_every = max(0, _env_int("ADALLM_INTERLEAVE", 0))
        # ADALLM_CHUNK_BUDGET: max tokens per prefill chunk per step.
        # 0 => legacy (chunk up to max_num_batched_tokens). When INTERLEAVE
        # is on without an explicit budget, default to 4096 so a waiting
        # decode waits at most one small chunk, not a 16K-token monster.
        _cb = _env_int("ADALLM_CHUNK_BUDGET", 0)
        if _cb > 0:
            self.chunk_budget = min(_cb, self.max_num_batched_tokens)
        elif self.interleave_every > 0 or _env_int("ADALLM_PARALLEL_PREFILL", 0) > 0:
            # small default budget so packed steps stay frequent enough for
            # piggybacked decode to progress (riders get 1 token per step)
            self.chunk_budget = min(4096, self.max_num_batched_tokens)
        else:
            self.chunk_budget = self.max_num_batched_tokens
        self._decode_steps_since_prefill = 1 << 30    # admit prefill initially
        # ---- Round5-B: parallel (packed) prefill (env-gated, default OFF) ----
        # ADALLM_PARALLEL_PREFILL=1:
        #   * multiple sequences' prompt chunks are packed into ONE prefill
        #     step (budget-shared continuations, no single-seq monopoly)
        #   * short prompts jump the queue: new seqs whose whole prompt fits
        #     the remaining budget are admitted shortest-first (SJF) while
        #     long-prompt chunks are in flight, with a reserved admission
        #     share so in-flight chunks still make progress
        #   * decode-phase seqs piggyback into prefill steps (1 token each,
        #     vLLM-style combined chunked-prefill+decode step), so decode
        #     never starves during long prefills
        # ADALLM_PIGGYBACK=0 disables only the decode piggyback (A/B knob).
        # ADALLM_PREFILL_RESERVE=T tokens reserved for new admissions when
        # chunked prefills are in flight (default 1024).
        self.parallel_prefill = _env_int("ADALLM_PARALLEL_PREFILL", 0) > 0
        self.prefill_piggyback = _env_int("ADALLM_PIGGYBACK", 1) > 0
        self.prefill_reserve = max(0, _env_int("ADALLM_PREFILL_RESERVE", 1024))
        # SJF starvation guard: if the FIFO head waits this many prefill
        # admission steps without being picked, it bypasses SJF entirely.
        self.sjf_max_wait = max(1, _env_int("ADALLM_SJF_MAX_WAIT", 8))
        self._sjf_head = None
        self._sjf_head_steps = 0
        # Round3-B instrumentation: decode batch-size histogram
        self.stats = {"prefill_steps": 0, "decode_steps": 0, "decode_bs_hist": {},
                      "packed_prefill_hist": {}, "sjf_admissions": 0,
                      "partial_admissions": 0, "piggyback_decode_tokens": 0}
        eos = config.eos
        if isinstance(eos, (list, tuple, set)):
            self.eos = set(eos)
        else:
            self.eos = eos
        # Hybrid models (gated delta-net layers) cannot reuse prefix KV cache:
        # the recurrent state summarizes the prefix and is never cached.
        layer_types = getattr(config.hf_config, "layer_types", None)
        is_hybrid = bool(layer_types) and any(t == "linear_attention" for t in layer_types)
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
            hybrid=config.hybrid_kv_cache,
            sink_blocks=config.kv_cache_sink_blocks,
            recent_blocks=config.kv_cache_recent_blocks,
            use_prefix_cache=not is_hybrid,
        )
        self.delta_slots = DeltaSlotManager(config.max_num_seqs + 2) if is_hybrid else None
        self.waiting: deque[Sequence] = deque()
        self.chunked: deque[Sequence] = deque()   # in-flight chunked prefills
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.chunked and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def _prefill_start(self, seq: Sequence) -> int:
        """Absolute token index where un-prefilled tokens begin."""
        return seq.num_cached_tokens + seq.num_prefilled_tokens

    def _prefill_remaining(self, seq: Sequence) -> int:
        return len(seq) - self._prefill_start(seq)

    def _prefill_admitted(self) -> bool:
        """Round3-B: when INTERLEAVE is on, skip prefill this step if decode
        seqs are running and we haven't run enough decode steps since the last
        prefill. Returns True when prefill may proceed this step."""
        if self.interleave_every <= 0 or not self.running:
            return True
        return self._decode_steps_since_prefill >= self.interleave_every

    def _schedule_parallel_prefill(self, scheduled: list[Sequence], step_budget: int) -> int:
        """Round5-B (ADALLM_PARALLEL_PREFILL=1): packed multi-seq prefill step.

        - In-flight chunked prefills share the budget (FIFO continuations),
          but at least `prefill_reserve` tokens of the budget are kept for
          NEW admissions while continuations run.
        - New admissions are shortest-first among seqs that fit ENTIRELY in
          the admission share (SJF whole-fit: short prompts jump the queue),
          then the shortest non-fit seq partially admitted into leftovers.
        - An aged FIFO head (starved by SJF for ADALLM_SJF_MAX_WAIT steps)
          bypasses SJF so long prompts cannot starve forever.
        - Decode-phase seqs piggyback into the prefill batch (1 token each,
        vLLM-style combined step) unless ADALLM_PIGGYBACK=0.
        """
        num_batched_tokens = 0

        # ---- continuations: shortest-remaining-first among in-flight chunks ----
        # (FIFO let one long prompt monopolize the budget and starve the
        # short prompts already mid-prefill behind it.)
        cont_cap = step_budget
        if self.chunked and self.waiting:
            cont_cap = max(step_budget - self.prefill_reserve, step_budget // 2)
        for seq in sorted(self.chunked, key=self._prefill_remaining):
            if num_batched_tokens >= cont_cap:
                break
            remaining = self._prefill_remaining(seq)
            take = min(remaining, cont_cap - num_batched_tokens)
            final = take == remaining
            seq.cur_chunk = take
            seq.chunk_final = final
            scheduled.append(seq)
            num_batched_tokens += take
            if final:
                self.chunked.remove(seq)
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)

        # ---- new admissions: SJF whole-fit, then partial shortest ----
        def _room_left() -> int:
            return step_budget - num_batched_tokens

        if self.waiting and _room_left() > 0:
            # starvation guard: an aged FIFO head bypasses SJF
            if self.waiting[0] is not self._sjf_head:
                self._sjf_head = self.waiting[0]
                self._sjf_head_steps = 0
            else:
                self._sjf_head_steps += 1
            aged = self._sjf_head_steps >= self.sjf_max_wait

            while (
                self.waiting
                and _room_left() > 0
                and len(scheduled) + len(self.running) + len(self.chunked) < self.max_num_seqs
            ):
                # candidates that fit whole, shortest first
                room = _room_left()
                fits = [
                    (self._prefill_remaining(s), i)
                    for i, s in enumerate(self.waiting)
                    if self._prefill_remaining(s) <= room
                ]
                if fits:
                    _, idx = min(fits)
                    self.stats["sjf_admissions"] += 1
                elif aged and self.block_manager.can_allocate(self.waiting[0]):
                    idx = 0    # starved long prompt: bypass SJF
                else:
                    # admit the SHORTEST non-fit seq partially into leftovers
                    rems = [self._prefill_remaining(s) for s in self.waiting]
                    idx = min(range(len(rems)), key=lambda i: rems[i])
                    if not self.block_manager.can_allocate(self.waiting[idx]):
                        break
                    self.stats["partial_admissions"] += 1
                seq = self.waiting[idx]
                if not self.block_manager.can_allocate(seq):
                    break
                remaining = self._prefill_remaining(seq)
                if remaining <= 0:
                    self.waiting.remove(seq)
                    continue
                self.block_manager.allocate(seq)
                if self.delta_slots is not None:
                    seq.delta_slot = self.delta_slots.alloc()
                take = min(remaining, _room_left())
                final = take == remaining
                seq.cur_chunk = take
                seq.chunk_final = final
                scheduled.append(seq)
                num_batched_tokens += take
                self.waiting.remove(seq)
                if final:
                    seq.status = SequenceStatus.RUNNING
                    self.running.append(seq)
                else:
                    seq.status = SequenceStatus.RUNNING
                    self.chunked.append(seq)
                if not fits and not aged:
                    break    # one partial admission per step, keep it simple

        # ---- decode piggyback: running seqs ride the prefill batch ----
        if self.prefill_piggyback and self.running and scheduled:
            sched_ids = {id(s) for s in scheduled}
            for seq in list(self.running):
                if len(scheduled) >= self.max_num_seqs:
                    break
                if id(seq) in sched_ids:
                    continue    # final chunk scheduled this step: token comes from it
                if not self.block_manager.can_append(seq):
                    break    # KV pressure: skip piggyback this step (safe)
                self.block_manager.may_append(seq)
                seq.cur_chunk = -1          # decode rider: 1 token
                seq.chunk_final = True
                scheduled.append(seq)
                self.stats["piggyback_decode_tokens"] += 1

        if len(scheduled) > 1:
            n = len(scheduled)
            self.stats["packed_prefill_hist"][n] = self.stats["packed_prefill_hist"].get(n, 0) + 1
        return num_batched_tokens

    def schedule(self) -> tuple[list[Sequence], bool]:
        # ---- chunked prefill: continue in-flight chunks, then admit new seqs ----
        scheduled: list[Sequence] = []
        num_batched_tokens = 0

        prefill_ok = self._prefill_admitted()
        # Round3-B: per-step chunk budget (legacy = max_num_batched_tokens).
        step_budget = self.chunk_budget

        if prefill_ok and (self.chunked or self.waiting):
            if self.parallel_prefill:
                num_batched_tokens = self._schedule_parallel_prefill(scheduled, step_budget)
            else:
                # Continue partially-prefilled sequences first (FIFO, whole budget).
                while self.chunked and num_batched_tokens < step_budget:
                    seq = self.chunked[0]
                    remaining = self._prefill_remaining(seq)
                    budget = step_budget - num_batched_tokens
                    take = min(remaining, budget)
                    final = take == remaining
                    seq.cur_chunk = take
                    seq.chunk_final = final
                    scheduled.append(seq)
                    num_batched_tokens += take
                    if final:
                        self.chunked.popleft()
                        seq.status = SequenceStatus.RUNNING
                        self.running.append(seq)
                    break    # one chunking seq per step (it eats the whole budget)

                # Admit new sequences while budget remains.
                while (
                    self.waiting
                    and num_batched_tokens < step_budget
                    and len(scheduled) + len(self.running) + len(self.chunked) < self.max_num_seqs
                ):
                    seq = self.waiting[0]
                    if not self.block_manager.can_allocate(seq):
                        break
                    remaining = self._prefill_remaining(seq)
                    if remaining <= 0:
                        self.waiting.popleft()
                        continue
                    self.block_manager.allocate(seq)
                    if self.delta_slots is not None:
                        seq.delta_slot = self.delta_slots.alloc()
                    budget = step_budget - num_batched_tokens
                    take = min(remaining, budget)
                    final = take == remaining
                    seq.cur_chunk = take
                    seq.chunk_final = final
                    scheduled.append(seq)
                    num_batched_tokens += take
                    self.waiting.popleft()
                    if final:
                        seq.status = SequenceStatus.RUNNING
                        self.running.append(seq)
                    else:
                        seq.status = SequenceStatus.RUNNING
                        self.chunked.append(seq)

        if scheduled:
            self._decode_steps_since_prefill = 0
            self.stats["prefill_steps"] += 1
            return scheduled, True

        self._decode_steps_since_prefill += 1

        # ---- decode ----
        if not self.running and (self.chunked or self.waiting):
            # Nothing schedulable right now (e.g. a prompt larger than the KV
            # pool): fail loudly with the actual reason instead of asserting.
            seq = (self.chunked or self.waiting)[0]
            need = seq.num_blocks
            free = len(self.block_manager.free_block_ids)
            raise RuntimeError(
                f"Cannot schedule: seq {seq.seq_id} needs {need} KV blocks, {free} free "
                f"(pool {len(self.block_manager.blocks)}, waiting {len(self.waiting)}, chunked {len(self.chunked)})"
            )
        num_seqs = 0
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                seq.cur_chunk = -1
                seq.chunk_final = True
                scheduled.append(seq)
        assert scheduled
        self.running.extendleft(reversed(scheduled))
        self.stats["decode_steps"] += 1
        self.stats["decode_bs_hist"][len(scheduled)] = self.stats["decode_bs_hist"].get(len(scheduled), 0) + 1
        return scheduled, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        if self.delta_slots is not None and seq.delta_slot:
            self.delta_slots.free(seq.delta_slot)
            seq.delta_slot = 0
        seq.num_prefilled_tokens = 0
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[tuple[int, int, bool]]:
        updates: list[tuple[int, int, bool]] = []
        for seq, token_id in zip(seqs, token_ids):
            if token_id == -1:
                # intermediate prefill chunk: advance the prefill cursor only
                seq.num_prefilled_tokens += seq.cur_chunk
                continue
            seq.append_token(token_id)
            finished = False
            if not seq.ignore_eos:
                if isinstance(self.eos, set):
                    finished = token_id in self.eos
                else:
                    finished = token_id == self.eos
            if finished or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                seq.finish_reason = "stop" if finished else "length"
                self.block_manager.deallocate(seq)
                if self.delta_slots is not None and seq.delta_slot:
                    self.delta_slots.free(seq.delta_slot)
                    seq.delta_slot = 0
                if seq in self.running:
                    self.running.remove(seq)
            updates.append((seq.seq_id, token_id, seq.status == SequenceStatus.FINISHED))
        return updates
