import torch
from torch import nn


def _maybe_compile(fn):
    """torch.compile optional: inductor's fork-pool compile workers deadlock
    when shapes change at runtime under a server (RTX 4090, Oct 2026)."""
    import os
    import functools
    if os.getenv("ADALLM_NO_COMPILE", ""):
        return fn
    return torch.compile(fn)



class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @_maybe_compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # temperature ~0 → greedy argmax (clamp avoids div-by-zero; the softmax
        # of huge logits is an exact one-hot, so the Gumbel-max still wins).
        temperatures = temperatures.clamp_min(1e-6)
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
