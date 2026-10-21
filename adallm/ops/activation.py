import torch
from torch import nn
import torch.nn.functional as F


def _maybe_compile(fn):
    """torch.compile optional: inductor's fork-pool compile workers deadlock
    when shapes change at runtime under a server (RTX 4090, Oct 2026)."""
    import os
    import functools
    if os.getenv("ADALLM_NO_COMPILE", ""):
        return fn
    return torch.compile(fn)



class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @_maybe_compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y


class GeluAndMul(nn.Module):

    def __init__(self, approximate: str = "tanh"):
        super().__init__()
        self.approximate = approximate

    @_maybe_compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.gelu(x, approximate=self.approximate) * y
