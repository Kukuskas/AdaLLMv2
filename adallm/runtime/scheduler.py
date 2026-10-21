from collections import deque

from adallm.settings import Config
from adallm.runtime.sequence import Sequence, SequenceStatus
from adallm.runtime.block_manager import BlockManager
from adallm.runtime.delta_slots import DeltaSlotManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
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

    def schedule(self) -> tuple[list[Sequence], bool]:
        # ---- chunked prefill: continue in-flight chunks, then admit new seqs ----
        scheduled: list[Sequence] = []
        num_batched_tokens = 0

        # Continue partially-prefilled sequences first (FIFO, whole budget).
        while self.chunked and num_batched_tokens < self.max_num_batched_tokens:
            seq = self.chunked[0]
            remaining = self._prefill_remaining(seq)
            budget = self.max_num_batched_tokens - num_batched_tokens
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
            and num_batched_tokens < self.max_num_batched_tokens
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
            budget = self.max_num_batched_tokens - num_batched_tokens
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
            return scheduled, True

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
            updates.append((seq.seq_id, token_id, finished))
        return updates
