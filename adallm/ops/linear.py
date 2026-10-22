import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from adallm.ops.nvfp4_gemm import nvfp4_linear
from adallm.infra.quant import get_quant_config
from adallm.ops import marlin as _marlin


def _linear_deq(module, x: torch.Tensor, wd: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """cuBLAS F.linear on a pre-dequantized weight (round2-d tiny-N path).

    If this module's weight was fused with a sibling (a+b / k+v), the shared
    fused weight produces both halves; slice out this module's rows.
    """
    if wd.dtype != x.dtype:
        wd = wd.to(x.dtype)
    out = F.linear(x, wd)
    if getattr(module, "fused_pair", False):
        n = module._deq_n
        if out.shape[-1] != n:
            off = module.fused_offset
            out = out[..., off:off + n]
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out


def pair_linear(mod_a: nn.Module, mod_b: nn.Module, x: torch.Tensor):
    """Run two sibling tiny-N linears; ONE fused F.linear when their
    dequantized weights were concatenated (ADALLM_TINYGEMM=1), else two
    normal module calls. Returns (out_a, out_b). Slice order follows each
    module's fused_offset — the cat order in _fuse_pairs need not match
    the argument order here."""
    wa = getattr(mod_a, "weight_deq", None)
    if wa is None or not getattr(mod_a, "fused_pair", False):
        return mod_a(x), mod_b(x)
    if wa.dtype != x.dtype:
        wa = wa.to(x.dtype)
    out = F.linear(x, wa)
    na = mod_a._deq_n
    off_a = getattr(mod_a, "fused_offset", 0)
    off_b = getattr(mod_b, "fused_offset", None)
    if off_b is None or getattr(mod_b, "weight_deq", None) is not wa:
        # sibling not fused with us: plain two calls
        return mod_a(x), mod_b(x)
    return out[..., off_a:off_a + na], out[..., off_b:off_b + mod_b._deq_n]

def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        qcfg = get_quant_config()
        self.nvfp4_enabled = qcfg.enabled
        self.nvfp4_mode = qcfg.mode
        self.nvfp4_group_size = qcfg.group_size
        if self.nvfp4_enabled and self.nvfp4_mode == "kernel":
            assert input_size % 2 == 0
            assert input_size % self.nvfp4_group_size == 0
            self.weight = nn.Parameter(
                torch.empty(output_size, input_size // 2, dtype=torch.uint8),
                requires_grad=False,
            )
            self.weight_scale = nn.Parameter(
                torch.empty(output_size, input_size // self.nvfp4_group_size, dtype=torch.float16),
                requires_grad=False,
            )
            # default 1.0 (no second-level scale); checkpoints that carry one
            # (ModelOpt weight_scale_2, QUASAR 1/global_scale) overwrite it.
            # QAT-style packs have no scale_2 — 1.0 is the correct identity.
            self.weight_scale_2 = nn.Parameter(torch.ones([], dtype=torch.float32), requires_grad=False)
            self.register_buffer("weight_scale_col", torch.empty(0), persistent=False)
            self.weight_scale.weight_loader = self.weight_loader
            self.weight_scale_2.weight_loader = self.scale_loader
        else:
            self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def scale_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, *_):
        param.data.copy_(loaded_weight)

    def _linear(self, x: torch.Tensor, bias_override: torch.Tensor | None = None) -> torch.Tensor:
        bias = self.bias if bias_override is None else bias_override
        if self.nvfp4_enabled and self.nvfp4_mode == "kernel":
            # tiny-N fast path: dequantized weight + cuBLAS (round2-d)
            wd = getattr(self, "weight_deq", None)
            if wd is not None and wd.numel() > 0:
                return _linear_deq(self, x, wd, bias)
            if _marlin.MARLIN_ENABLED:
                out = _marlin.marlin_linear(self, x, bias)
                if out is not None:
                    return out
                # fallback: Triton path on original weights (only present
                # when ADALLM_MARLIN_FREE_TRITON=0, or repack failed)
                w = getattr(self, "weight_triton", None)
                ws = getattr(self, "weight_scale_triton", None)
                if w is not None and ws is not None:
                    w_scale_col = self.weight_scale_col if self.weight_scale_col.numel() else None
                    return nvfp4_linear(
                        x, w, ws, self.weight_scale_2, w_scale_col, bias,
                        self.nvfp4_group_size,
                    )
            w_scale_col = self.weight_scale_col if self.weight_scale_col.numel() else None
            return nvfp4_linear(
                x,
                self.weight,
                self.weight_scale,
                self.weight_scale_2,
                w_scale_col,
                bias,
                self.nvfp4_group_size,
            )
        return F.linear(x, self.weight, bias)


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._linear(x)


class FP16Linear(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__()
        dtype = torch.get_default_dtype()
        self.weight = nn.Parameter(torch.empty(output_size, input_size, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._linear(x)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._linear(x, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
