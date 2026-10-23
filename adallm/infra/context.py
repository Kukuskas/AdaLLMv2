from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    delta_slots: torch.Tensor | None = None    # [N] int64 state-pool slot per sequence
    # Round8-C: host mirrors (Python ints) for graph-capture-unsafe host reads.
    # Populated by prepare_prefill/prepare_decode which build them from lists.
    cu_q_list: list | None = None              # [n_seqs+1] chunk-relative q offsets
    slots_list: list | None = None             # [n_seqs] delta slot ids
    q_len_list: list | None = None             # [n_seqs] chunk lengths

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    delta_slots=None,
    cu_q_list=None,
    slots_list=None,
    q_len_list=None,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables,
        delta_slots,
        cu_q_list,
        slots_list,
        q_len_list,
    )

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
