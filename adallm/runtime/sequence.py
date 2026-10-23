from copy import copy
from enum import Enum, auto
from itertools import count

from adallm.sampling import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        # delta-net state pool slot (0 = NULL/unassigned until allocated)
        self.delta_slot = 0
        # chunked prefill bookkeeping
        self.num_prefilled_tokens = 0
        self.cur_chunk = -1
        self.chunk_final = True
        self.finish_reason: str | None = None    # 'stop' | 'length' when done

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        # Round8-D (TP=2): pickle the FULL dynamic state — rank-1 workers
        # rebuild sequences from these pickles and their prepare_prefill /
        # prepare_decode / postprocess need every field. The old 5-tuple
        # predated chunked prefill, delta slots, and finish bookkeeping;
        # each missing field crashed rank-1 with AttributeError.
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token,
                self.num_prefilled_tokens, self.cur_chunk, self.chunk_final,
                self.delta_slot, self.seq_id, self.status.value,
                self.temperature, self.max_tokens, self.ignore_eos, self.finish_reason)

    def __setstate__(self, state):
        (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
         tok_or_last, self.num_prefilled_tokens, self.cur_chunk, self.chunk_final,
         self.delta_slot, self.seq_id, status, self.temperature, self.max_tokens,
         self.ignore_eos, self.finish_reason) = state
        self.status = SequenceStatus(status)
        if self.num_completion_tokens == 0:
            self.token_ids = tok_or_last
            self.last_token = self.token_ids[-1] if self.token_ids else -1
        else:
            self.last_token = tok_or_last
