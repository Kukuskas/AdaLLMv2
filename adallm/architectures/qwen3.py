import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from collections import OrderedDict
import time
from transformers import Qwen3Config, Qwen3NextConfig

from adallm.ops.activation import SiluAndMul
from adallm.ops.attention import Attention
from adallm.ops.layernorm import RMSNorm
from adallm.ops.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear, FP16Linear, ReplicatedLinear
from adallm.ops.rotary_embedding import get_rope
from adallm.ops.embed_head import VocabParallelEmbedding, ParallelLMHead
from adallm.ops.nvfp4_gemm import nvfp4_linear, nvfp4_linear_dual, nvfp4_linear_gather
from adallm.infra.quant import get_quant_config
from adallm.infra.context import get_context

try:
    from vllm.model_executor.layers.fla.ops import fused_recurrent_gated_delta_rule
    _HAS_VLLM_GDN = True
except Exception:
    fused_recurrent_gated_delta_rule = None
    _HAS_VLLM_GDN = False


class Qwen3NextRMSNormGated(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states * self.weight.float()
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


class Qwen3NextRMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float()
        output = output * torch.rsqrt(output.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_next(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    rotary_dim = cos.shape[-1]
    if q.dim() == 3:
        if cos.dim() == 3 and cos.shape[0] == 1:
            cos = cos[0]
            sin = sin[0]
        cos = cos[:, None, :]
        sin = sin[:, None, :]
    elif q.dim() == 4:
        if cos.dim() == 2:
            cos = cos[None, :, :]
            sin = sin[None, :, :]
        cos = cos[:, :, None, :]
        sin = sin[:, :, None, :]
    else:
        raise ValueError(f"Unsupported q dim for rotary: {q.dim()}")
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


class Qwen3NextRotaryEmbedding(nn.Module):

    def __init__(self, config: Qwen3NextConfig, max_position: int | None = None):
        super().__init__()
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        partial = getattr(config, "partial_rotary_factor", 1.0)
        rotary_dim = int(head_dim * partial)
        rotary_dim = max(2, rotary_dim - (rotary_dim % 2))
        base = getattr(config, "rope_theta", 1000000)
        if max_position is None:
            max_position = config.max_position_embeddings
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.dim() == 0:
            positions = positions.view(1)
        elif positions.dim() > 1:
            positions = positions.reshape(-1)
        cos = self.cos_cache.index_select(0, positions)
        sin = self.sin_cache.index_select(0, positions)
        return cos.to(dtype=self.cos_cache.dtype), sin.to(dtype=self.sin_cache.dtype)
def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool = True,
    use_qk_l2norm: bool = True,
):
    initial_dtype = query.dtype
    if use_qk_l2norm:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim, device=value.device, dtype=value.dtype)
    if initial_state is None:
        last_recurrent_state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device, dtype=value.dtype)
    else:
        last_recurrent_state = initial_state.to(value.dtype)

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class Qwen3NextGatedDeltaNet(nn.Module):

    def __init__(self, config: Qwen3NextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        projection_size_ba = self.num_v_heads * 2
        self.in_proj_qkvz = FP16Linear(self.hidden_size, projection_size_qkvz, bias=False)
        self.in_proj_ba = FP16Linear(self.hidden_size, projection_size_ba, bias=False)

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.norm = Qwen3NextRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.out_proj = RowParallelLinear(self.value_dim, self.hidden_size, bias=False)

        self._conv_state: torch.Tensor | None = None
        self._recurrent_state: torch.Tensor | None = None

    def fix_query_key_value_ordering(self, mixed_qkvz: torch.Tensor, mixed_ba: torch.Tensor):
        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads,
            2 * self.head_k_dim + 2 * self.head_v_dim * self.num_v_heads // self.num_k_heads,
        )
        new_tensor_shape_ba = mixed_ba.size()[:-1] + (self.num_k_heads, 2 * self.num_v_heads // self.num_k_heads)

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)
        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [self.num_v_heads // self.num_k_heads, self.num_v_heads // self.num_k_heads]
        query, key, value, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=3)
        b, a = torch.split(mixed_ba, split_arg_list_ba, dim=3)
        value = value.reshape(value.size(0), value.size(1), -1, self.head_v_dim)
        z = z.reshape(z.size(0), z.size(1), -1, self.head_v_dim)
        b = b.reshape(b.size(0), b.size(1), self.num_v_heads)
        a = a.reshape(a.size(0), a.size(1), self.num_v_heads)
        return query, key, value, z, b, a

    def _conv_update(self, mixed_qkv_t: torch.Tensor) -> torch.Tensor:
        if self.conv_kernel_size <= 1:
            return F.silu(mixed_qkv_t)
        if self._conv_state is None or self._conv_state.shape[0] != mixed_qkv_t.shape[0]:
            self._conv_state = mixed_qkv_t.new_zeros(mixed_qkv_t.shape[0], self.conv_dim, self.conv_kernel_size - 1)
        window = torch.cat([self._conv_state, mixed_qkv_t], dim=-1)
        weight = self.conv1d.weight.squeeze(1).to(window.dtype)
        out = (window * weight.unsqueeze(0)).sum(dim=-1, keepdim=True)
        out = F.silu(out)
        self._conv_state = window[:, :, 1:].detach()
        return out

    def _update_conv_state_prefill(self, mixed_qkv_t: torch.Tensor, seq_len: int):
        if self.conv_kernel_size <= 1:
            self._conv_state = None
            return
        if seq_len >= self.conv_kernel_size - 1:
            self._conv_state = mixed_qkv_t[:, :, -(self.conv_kernel_size - 1):].detach()
        else:
            pad = self.conv_kernel_size - 1 - seq_len
            zeros = mixed_qkv_t.new_zeros(mixed_qkv_t.shape[0], self.conv_dim, pad)
            self._conv_state = torch.cat([zeros, mixed_qkv_t], dim=-1).detach()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_context()
        squeeze_output = False
        input_dtype = hidden_states.dtype
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
            squeeze_output = True
        batch_size, seq_len, _ = hidden_states.shape

        if context.is_prefill:
            self._conv_state = None
            self._recurrent_state = None

        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(projected_states_qkvz, projected_states_ba)
        query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))

        mixed_qkv = torch.cat((query, key, value), dim=-1)
        mixed_qkv_t = mixed_qkv.transpose(1, 2)

        if context.is_prefill or seq_len > 1:
            conv_out = self.conv1d(mixed_qkv_t)
            conv_out = conv_out[:, :, :seq_len]
            conv_out = F.silu(conv_out)
            self._update_conv_state_prefill(mixed_qkv_t, seq_len)
            mixed_qkv_conv = conv_out.transpose(1, 2)
            initial_state = None
        else:
            conv_out = self._conv_update(mixed_qkv_t)
            mixed_qkv_conv = conv_out.transpose(1, 2)
            initial_state = self._recurrent_state

        query, key, value = torch.split(
            mixed_qkv_conv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = torch.sigmoid(b)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            repeat = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)

        if _HAS_VLLM_GDN and query.is_cuda:
            if initial_state is None:
                initial_state = torch.zeros(
                    query.shape[0],
                    value.shape[2],
                    self.head_k_dim,
                    self.head_v_dim,
                    device=value.device,
                    dtype=value.dtype,
                )
            core_attn_out, last_state = fused_recurrent_gated_delta_rule(
                q=query.contiguous(),
                k=key.contiguous(),
                v=value.contiguous(),
                g=g.contiguous(),
                beta=beta.contiguous(),
                initial_state=initial_state,
                inplace_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_state = _recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm=True,
            )
        self._recurrent_state = last_state.detach() if last_state is not None else None

        z_shape = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)

        output = self.out_proj(core_attn_out)
        if output.dtype != input_dtype:
            output = output.to(input_dtype)
        if squeeze_output:
            output = output.squeeze(0)
        return output


class Qwen3NextAttention(nn.Module):

    def __init__(self, config: Qwen3NextConfig):
        super().__init__()
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = FP16Linear(config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=config.attention_bias)
        self.k_proj = FP16Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = FP16Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = RowParallelLinear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        max_position = min(
            config.max_position_embeddings,
            int(getattr(config, "max_model_len", config.max_position_embeddings)),
        )
        self.rotary_emb = Qwen3NextRotaryEmbedding(config, max_position=max_position)
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        q_out = self.q_proj(hidden_states)
        q_out = q_out.view(-1, self.num_heads, self.head_dim * 2)
        q, gate = q_out.split(self.head_dim, dim=-1)
        gate = gate.reshape(-1, self.num_heads * self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(self.k_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim))
        v = self.v_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        cos, sin = self.rotary_emb(positions)
        if cos.dim() == 3 and q.dim() == 3 and cos.shape[0] == 1:
            cos = cos[0]
            sin = sin[0]
        if cos.dtype != q.dtype:
            cos = cos.to(q.dtype)
            sin = sin.to(q.dtype)
        q, k = _apply_rotary_pos_emb_next(q, k, cos, sin)
        o = self.attn(q, k, v)
        o = o.flatten(1, -1)
        o = o * torch.sigmoid(gate)
        return self.o_proj(o)


class Qwen3NextSharedExpert(nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = ReplicatedLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = ReplicatedLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = ReplicatedLinear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class _OffloadLinearStorage(nn.Module):

    def __init__(self, in_features: int, out_features: int, group_size: int, device: torch.device, pin_memory: bool):
        super().__init__()
        self.group_size = group_size
        self.register_buffer(
            "weight",
            torch.empty(out_features, in_features // 2, dtype=torch.uint8, device=device),
            persistent=False,
        )
        self.register_buffer(
            "weight_scale",
            torch.empty(out_features, in_features // group_size, dtype=torch.float16, device=device),
            persistent=False,
        )
        self.register_buffer(
            "weight_scale_2",
            torch.ones([], dtype=torch.float32, device=device),
            persistent=False,
        )
        if pin_memory and device.type == "cpu":
            self._buffers["weight"] = self.weight.pin_memory()
            self._buffers["weight_scale"] = self.weight_scale.pin_memory()
            self._buffers["weight_scale_2"] = self.weight_scale_2.pin_memory()

    def to_gpu(self, device: torch.device):
        if self.weight.device == device:
            return self.weight, self.weight_scale, self.weight_scale_2
        return (
            self.weight.to(device, non_blocking=True),
            self.weight_scale.to(device, non_blocking=True),
            self.weight_scale_2.to(device, non_blocking=True),
        )


class _MoEExpert(nn.Module):

    def __init__(self, config, group_size: int, device: torch.device, pin_memory: bool):
        super().__init__()
        hidden_size = config.hidden_size
        moe_intermediate = config.moe_intermediate_size
        self.gate_proj = _OffloadLinearStorage(hidden_size, moe_intermediate, group_size, device, pin_memory)
        self.up_proj = _OffloadLinearStorage(hidden_size, moe_intermediate, group_size, device, pin_memory)
        self.down_proj = _OffloadLinearStorage(moe_intermediate, hidden_size, group_size, device, pin_memory)

    def to_gpu(self, device: torch.device):
        g = self.gate_proj.to_gpu(device)
        u = self.up_proj.to_gpu(device)
        d = self.down_proj.to_gpu(device)
        return g + u + d


class _SharedLinearView(nn.Module):

    def __init__(self, weight: torch.Tensor, weight_scale: torch.Tensor, weight_scale_2: torch.Tensor, idx: int):
        super().__init__()
        self.register_buffer("weight", weight[idx], persistent=False)
        self.register_buffer("weight_scale", weight_scale[idx], persistent=False)
        self.register_buffer("weight_scale_2", weight_scale_2[idx], persistent=False)


class _MoEExpertView(nn.Module):

    def __init__(
        self,
        gate_weight: torch.Tensor,
        gate_scale: torch.Tensor,
        gate_scale_2: torch.Tensor,
        up_weight: torch.Tensor,
        up_scale: torch.Tensor,
        up_scale_2: torch.Tensor,
        down_weight: torch.Tensor,
        down_scale: torch.Tensor,
        down_scale_2: torch.Tensor,
        idx: int,
    ):
        super().__init__()
        self.gate_proj = _SharedLinearView(gate_weight, gate_scale, gate_scale_2, idx)
        self.up_proj = _SharedLinearView(up_weight, up_scale, up_scale_2, idx)
        self.down_proj = _SharedLinearView(down_weight, down_scale, down_scale_2, idx)


class _MoEExpertCache:

    def __init__(self, capacity: int, decay: float = 0.95, decay_interval: int = 128):
        self.capacity = max(int(capacity), 0)
        self.cache: OrderedDict[int, tuple[torch.Tensor, ...]] = OrderedDict()
        self.freq: dict[int, float] = {}
        self.step = 0
        self.decay = float(decay)
        self.decay_interval = int(decay_interval)
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _maybe_decay(self):
        if self.decay_interval <= 0:
            return
        if self.step % self.decay_interval != 0:
            return
        for key in list(self.freq.keys()):
            self.freq[key] *= self.decay

    def _evict_one(self):
        if not self.cache:
            return
        min_freq = min(self.freq.get(k, 0.0) for k in self.cache.keys())
        for k in list(self.cache.keys()):
            if self.freq.get(k, 0.0) == min_freq:
                self.cache.pop(k, None)
                self.freq.pop(k, None)
                self.evictions += 1
                return

    def get(
        self,
        expert_id: int,
        expert: _MoEExpert,
        device: torch.device,
        count: int = 1,
        update_freq: bool = True,
        touch: bool = True,
    ):
        if self.capacity <= 0:
            return expert.to_gpu(device)
        self.step += 1
        self._maybe_decay()
        if expert_id in self.cache:
            weights = self.cache[expert_id]
            if touch:
                self.cache.move_to_end(expert_id)
            self.hits += 1
        else:
            if len(self.cache) >= self.capacity:
                self._evict_one()
            weights = expert.to_gpu(device)
            self.cache[expert_id] = weights
            self.misses += 1
        if update_freq:
            self.freq[expert_id] = self.freq.get(expert_id, 0.0) + float(count)
        return weights

    def get_cached(self, expert_id: int, count: int = 1, touch: bool = True):
        if expert_id not in self.cache:
            return None
        weights = self.cache[expert_id]
        if touch:
            self.cache.move_to_end(expert_id)
        self.hits += 1
        self.freq[expert_id] = self.freq.get(expert_id, 0.0) + float(count)
        return weights

    def insert(self, expert_id: int, weights: tuple[torch.Tensor, ...], count: int = 1, touch: bool = True):
        if self.capacity <= 0:
            return
        if expert_id in self.cache:
            self.cache[expert_id] = weights
            if touch:
                self.cache.move_to_end(expert_id)
        else:
            if len(self.cache) >= self.capacity:
                self._evict_one()
            self.cache[expert_id] = weights
            self.misses += 1
        self.freq[expert_id] = self.freq.get(expert_id, 0.0) + float(count)

    def stats(self) -> tuple[int, int, int]:
        return self.hits, self.misses, self.evictions


class Qwen3MoeSparseMoeBlock(nn.Module):

    def __init__(self, config, layer_idx: int | None = None):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = FP16Linear(config.hidden_size, config.num_experts, bias=False)
        self.layer_idx = int(layer_idx) if layer_idx is not None else -1

        qcfg = get_quant_config()
        if not qcfg.enabled or qcfg.mode != "kernel":
            raise RuntimeError("MoE expert weights require NVFP4 kernel mode.")
        group_size = qcfg.group_size

        offload = bool(getattr(config, "moe_offload_experts", True))
        cache_capacity = int(getattr(config, "moe_gpu_expert_capacity", self.top_k))
        pin_memory = bool(getattr(config, "moe_offload_pin_memory", False))
        self.prefetch = bool(getattr(config, "moe_offload_prefetch", False))
        self.prefetch_window = int(getattr(config, "moe_offload_prefetch_window", 0))
        self._prefetch_stream = torch.cuda.Stream() if offload and self.prefetch else None
        if offload:
            storage_device = torch.device("cpu")
            self.experts = nn.ModuleList(
                [_MoEExpert(config, group_size, storage_device, pin_memory) for _ in range(self.num_experts)]
            )
            decay = float(getattr(config, "moe_cache_decay", 0.95))
            decay_interval = int(getattr(config, "moe_cache_decay_interval", 128))
            self.expert_cache = _MoEExpertCache(cache_capacity, decay=decay, decay_interval=decay_interval)
            self.pack_enabled = bool(getattr(config, "moe_offload_pack", True))
            self.pack_cache = bool(getattr(config, "moe_offload_pack_cache", True))
            self.pack_reuse = bool(getattr(config, "moe_offload_pack_reuse", True))
            pack_capacity = int(getattr(config, "moe_offload_pack_capacity", 0))
            self.pack_capacity = max(pack_capacity, self.top_k)
            self.pack_gather_min_rows = int(getattr(config, "moe_offload_pack_gather_min_rows", 128))
            if self.pack_enabled:
                hidden = config.hidden_size
                inter = config.moe_intermediate_size
                g_w_numel = inter * (hidden // 2)
                u_w_numel = inter * (hidden // 2)
                d_w_numel = hidden * (inter // 2)
                g_s_numel = inter * (hidden // group_size)
                u_s_numel = inter * (hidden // group_size)
                d_s_numel = hidden * (inter // group_size)
                self._pack_w_numel = g_w_numel + u_w_numel + d_w_numel
                self._pack_s_numel = g_s_numel + u_s_numel + d_s_numel + 3
                self._pack_w_offsets = (0, g_w_numel, g_w_numel + u_w_numel, self._pack_w_numel)
                self._pack_s_offsets = (0, g_s_numel, g_s_numel + u_s_numel, g_s_numel + u_s_numel + d_s_numel)
                self._pack_s2_offset = self._pack_s_offsets[-1]
                self._pack_shapes = (
                    (inter, hidden // 2),
                    (inter, hidden // 2),
                    (hidden, inter // 2),
                    (inter, hidden // group_size),
                    (inter, hidden // group_size),
                    (hidden, inter // group_size),
                )
                self._pack_cpu_w = torch.empty(
                    (self.pack_capacity, self._pack_w_numel),
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=pin_memory,
                )
                self._pack_cpu_s = torch.empty(
                    (self.pack_capacity, self._pack_s_numel),
                    dtype=torch.float16,
                    device="cpu",
                    pin_memory=pin_memory,
                )
                self._pack_gpu_w = torch.empty(
                    (self.pack_capacity, self._pack_w_numel),
                    dtype=torch.uint8,
                    device="cuda",
                )
                self._pack_gpu_s = torch.empty(
                    (self.pack_capacity, self._pack_s_numel),
                    dtype=torch.float16,
                    device="cuda",
                )
                if self.pack_reuse:
                    self._pack_cache_decay = decay
                    self._pack_cache_decay_interval = decay_interval
                    self._pack_cache_step = 0
                    self._pack_row_to_expert = [-1 for _ in range(self.pack_capacity)]
                    self._pack_expert_to_row: dict[int, int] = {}
                    self._pack_row_score = [0.0 for _ in range(self.pack_capacity)]
                    self._pack_row_age = [0 for _ in range(self.pack_capacity)]
        else:
            storage_device = torch.device("cuda")
            hidden_size = config.hidden_size
            moe_intermediate = config.moe_intermediate_size
            self.register_buffer(
                "_gate_weight",
                torch.empty((self.num_experts, moe_intermediate, hidden_size // 2), dtype=torch.uint8, device=storage_device),
                persistent=False,
            )
            self.register_buffer(
                "_gate_scale",
                torch.empty((self.num_experts, moe_intermediate, hidden_size // group_size), dtype=torch.float16, device=storage_device),
                persistent=False,
            )
            self.register_buffer(
                "_gate_scale_2",
                torch.empty((self.num_experts,), dtype=torch.float32, device=storage_device),
                persistent=False,
            )
            self.register_buffer(
                "_up_weight",
                torch.empty((self.num_experts, moe_intermediate, hidden_size // 2), dtype=torch.uint8, device=storage_device),
                persistent=False,
            )
            self.register_buffer(
                "_up_scale",
                torch.empty((self.num_experts, moe_intermediate, hidden_size // group_size), dtype=torch.float16, device=storage_device),
                persistent=False,
            )
            self.register_buffer(
                "_up_scale_2",
                torch.empty((self.num_experts,), dtype=torch.float32, device=storage_device),
                persistent=False,
            )
            self.register_buffer(
                "_down_weight",
                torch.empty((self.num_experts, hidden_size, moe_intermediate // 2), dtype=torch.uint8, device=storage_device),
                persistent=False,
            )
            self.register_buffer(
                "_down_scale",
                torch.empty((self.num_experts, hidden_size, moe_intermediate // group_size), dtype=torch.float16, device=storage_device),
                persistent=False,
            )
            self.register_buffer(
                "_down_scale_2",
                torch.empty((self.num_experts,), dtype=torch.float32, device=storage_device),
                persistent=False,
            )
            self.experts = nn.ModuleList(
                [
                    _MoEExpertView(
                        self._gate_weight,
                        self._gate_scale,
                        self._gate_scale_2,
                        self._up_weight,
                        self._up_scale,
                        self._up_scale_2,
                        self._down_weight,
                        self._down_scale,
                        self._down_scale_2,
                        idx,
                    )
                    for idx in range(self.num_experts)
                ]
            )
            self.expert_cache = None
            self.pack_enabled = False
            self.pack_cache = False
            self.pack_reuse = False
            self.pack_capacity = 0
            self.pack_gather_min_rows = 0
        self.group_size = group_size
        self.offload = offload
        self.profile_enabled = bool(getattr(config, "moe_profile", False))
        self.profile_interval = int(getattr(config, "moe_profile_interval", 1))
        profile_layers = getattr(config, "moe_profile_layers", None)
        if self.profile_enabled:
            if profile_layers is None:
                self.profile_enabled = self.layer_idx == 0
            elif isinstance(profile_layers, (list, tuple)):
                self.profile_enabled = self.layer_idx in profile_layers
            elif isinstance(profile_layers, int):
                self.profile_enabled = self.layer_idx == profile_layers
            else:
                self.profile_enabled = False
        self._profile_step = 0
        self._last_cache_stats = (0, 0, 0)
        self._expert_bytes = self._estimate_expert_bytes(config.hidden_size, config.moe_intermediate_size, group_size)
        if self.offload and self.pack_enabled:
            w_bytes = self._pack_w_numel
            s_bytes = self._pack_s_numel * 2
            self._expert_bytes = int(w_bytes + s_bytes)

    @staticmethod
    def _estimate_expert_bytes(hidden_size: int, moe_intermediate: int, group_size: int) -> int:
        w_bytes = (
            moe_intermediate * (hidden_size // 2)
            + moe_intermediate * (hidden_size // 2)
            + hidden_size * (moe_intermediate // 2)
        )
        s_bytes = (
            moe_intermediate * (hidden_size // group_size)
            + moe_intermediate * (hidden_size // group_size)
            + hidden_size * (moe_intermediate // group_size)
        ) * 2
        s2_bytes = 3 * 4
        return int(w_bytes + s_bytes + s2_bytes)

    def _expert_forward(self, x: torch.Tensor, weights: tuple[torch.Tensor, ...]) -> torch.Tensor:
        g_w, g_s, g_s2, u_w, u_s, u_s2, d_w, d_s, d_s2 = weights
        gate, up = nvfp4_linear_dual(
            x,
            g_w,
            g_s,
            g_s2,
            u_w,
            u_s,
            u_s2,
            None,
            None,
            self.group_size,
        )
        act = F.silu(gate) * up
        return nvfp4_linear(act, d_w, d_s, d_s2, None, None, self.group_size)

    def _pack_cpu(self, expert_ids: list[int], rows: list[int] | None = None) -> int:
        num = len(expert_ids)
        if num == 0:
            return 0
        if rows is None:
            rows = list(range(num))
        if len(rows) != num:
            raise ValueError("rows must match expert_ids length")
        if max(rows) >= self.pack_capacity:
            raise RuntimeError("Packed MoE buffer too small for requested experts.")
        w0, w1, w2, w3 = self._pack_w_offsets
        s0, s1, s2, _ = self._pack_s_offsets
        s2_off = self._pack_s2_offset
        for row_idx, expert_id in zip(rows, expert_ids):
            expert = self.experts[expert_id]
            g_w = expert.gate_proj.weight
            g_s = expert.gate_proj.weight_scale
            g_s2 = expert.gate_proj.weight_scale_2
            u_w = expert.up_proj.weight
            u_s = expert.up_proj.weight_scale
            u_s2 = expert.up_proj.weight_scale_2
            d_w = expert.down_proj.weight
            d_s = expert.down_proj.weight_scale
            d_s2 = expert.down_proj.weight_scale_2
            row_w = self._pack_cpu_w[row_idx]
            row_s = self._pack_cpu_s[row_idx]
            row_w[w0:w1].copy_(g_w.view(-1))
            row_w[w1:w2].copy_(u_w.view(-1))
            row_w[w2:w3].copy_(d_w.view(-1))
            row_s[s0:s1].copy_(g_s.view(-1))
            row_s[s1:s2].copy_(u_s.view(-1))
            row_s[s2:s2_off].copy_(d_s.view(-1))
            row_s[s2_off:s2_off + 1].copy_(g_s2.to(torch.float16).view(1))
            row_s[s2_off + 1:s2_off + 2].copy_(u_s2.to(torch.float16).view(1))
            row_s[s2_off + 2:s2_off + 3].copy_(d_s2.to(torch.float16).view(1))
        return num

    def _unpack_gpu(self, idx: int) -> tuple[torch.Tensor, ...]:
        g_w_shape, u_w_shape, d_w_shape, g_s_shape, u_s_shape, d_s_shape = self._pack_shapes
        w0, w1, w2, w3 = self._pack_w_offsets
        s0, s1, s2, _ = self._pack_s_offsets
        s2_off = self._pack_s2_offset
        row_w = self._pack_gpu_w[idx]
        row_s = self._pack_gpu_s[idx]
        g_w = row_w[w0:w1].view(g_w_shape)
        u_w = row_w[w1:w2].view(u_w_shape)
        d_w = row_w[w2:w3].view(d_w_shape)
        g_s = row_s[s0:s1].view(g_s_shape)
        u_s = row_s[s1:s2].view(u_s_shape)
        d_s = row_s[s2:s2_off].view(d_s_shape)
        g_s2 = row_s[s2_off:s2_off + 1]
        u_s2 = row_s[s2_off + 1:s2_off + 2]
        d_s2 = row_s[s2_off + 2:s2_off + 3]
        return g_w, g_s, g_s2, u_w, u_s, u_s2, d_w, d_s, d_s2

    @staticmethod
    def _clone_weights(weights: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return tuple(w.clone() for w in weights)

    def _pack_gpu_from_weights(self, row_idx: int, weights: tuple[torch.Tensor, ...]) -> None:
        g_w, g_s, g_s2, u_w, u_s, u_s2, d_w, d_s, d_s2 = weights
        w0, w1, w2, w3 = self._pack_w_offsets
        s0, s1, s2, _ = self._pack_s_offsets
        s2_off = self._pack_s2_offset
        row_w = self._pack_gpu_w[row_idx]
        row_s = self._pack_gpu_s[row_idx]
        row_w[w0:w1].copy_(g_w.view(-1))
        row_w[w1:w2].copy_(u_w.view(-1))
        row_w[w2:w3].copy_(d_w.view(-1))
        row_s[s0:s1].copy_(g_s.view(-1))
        row_s[s1:s2].copy_(u_s.view(-1))
        row_s[s2:s2_off].copy_(d_s.view(-1))
        row_s[s2_off:s2_off + 1].copy_(g_s2.to(torch.float16).view(1))
        row_s[s2_off + 1:s2_off + 2].copy_(u_s2.to(torch.float16).view(1))
        row_s[s2_off + 2:s2_off + 3].copy_(d_s2.to(torch.float16).view(1))

    def _pack_cache_maybe_decay(self) -> None:
        if not self.pack_reuse:
            return
        if self._pack_cache_decay_interval <= 0:
            return
        if self._pack_cache_step % self._pack_cache_decay_interval != 0:
            return
        for idx in range(self.pack_capacity):
            self._pack_row_score[idx] *= self._pack_cache_decay

    def _pack_cache_alloc_row(self, expert_id: int) -> int:
        row = -1
        for idx, cached_id in enumerate(self._pack_row_to_expert):
            if cached_id < 0:
                row = idx
                break
        if row < 0:
            row = min(
                range(self.pack_capacity),
                key=lambda i: (self._pack_row_score[i], self._pack_row_age[i]),
            )
            old_id = self._pack_row_to_expert[row]
            if old_id >= 0:
                self._pack_expert_to_row.pop(old_id, None)
        self._pack_row_to_expert[row] = expert_id
        self._pack_expert_to_row[expert_id] = row
        self._pack_row_score[row] = 0.0
        self._pack_row_age[row] = self._pack_cache_step
        return row

    def _pack_cache_touch(self, row: int, count: int) -> None:
        self._pack_row_score[row] += float(count)
        self._pack_row_age[row] = self._pack_cache_step

    def forward(self, hidden_states: torch.Tensor):
        orig_shape = hidden_states.shape
        if hidden_states.dim() == 2:
            hidden_dim = hidden_states.shape[-1]
        else:
            hidden_dim = hidden_states.shape[-1]
            hidden_states = hidden_states.view(-1, hidden_dim)
        if hidden_states.dtype != self.gate.weight.dtype:
            hidden_states = hidden_states.to(self.gate.weight.dtype)
        do_profile = self.profile_enabled and (self._profile_step % self.profile_interval == 0)
        self._profile_step += 1
        if do_profile:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        router_logits = self.gate(hidden_states)

        routing_weights = torch.softmax(router_logits, dim=1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        if do_profile:
            torch.cuda.synchronize()
            t_route = (time.perf_counter() - t0) * 1000.0

        final_hidden_states = torch.zeros_like(hidden_states)
        if self.offload:
            flat_experts = selected_experts.reshape(-1)
            counts = torch.bincount(flat_experts, minlength=self.num_experts).to(torch.int64).cpu()
            counts_list = counts.tolist()
            unique_experts = [i for i, c in enumerate(counts_list) if c > 0]
            unique_experts.sort(key=lambda i: counts_list[i], reverse=True)
            if do_profile:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            use_pack = self.pack_enabled and len(unique_experts) <= self.pack_capacity
            use_pack_gather = use_pack and (hidden_states.shape[0] * self.top_k >= self.pack_gather_min_rows)
            if use_pack_gather:
                num = len(unique_experts)
                cached_rows: list[tuple[int, tuple[torch.Tensor, ...]]] = []
                cpu_rows: list[int] = []
                cpu_ids: list[int] = []
                row_for_expert: dict[int, int] = {}
                if self.pack_reuse:
                    self._pack_cache_step += 1
                    self._pack_cache_maybe_decay()
                    for expert_id in unique_experts:
                        row_idx = self._pack_expert_to_row.get(expert_id, -1)
                        if row_idx >= 0:
                            row_for_expert[expert_id] = row_idx
                            continue
                        row_idx = self._pack_cache_alloc_row(expert_id)
                        row_for_expert[expert_id] = row_idx
                        cached = None
                        if self.expert_cache is not None:
                            cached = self.expert_cache.get_cached(
                                expert_id,
                                count=int(counts_list[expert_id]),
                                touch=True,
                            )
                        if cached is None:
                            cpu_rows.append(row_idx)
                            cpu_ids.append(expert_id)
                        else:
                            cached_rows.append((row_idx, cached))
                    for expert_id in unique_experts:
                        row_idx = row_for_expert[expert_id]
                        self._pack_cache_touch(row_idx, counts_list[expert_id])
                else:
                    if self.expert_cache is not None:
                        for row_idx, expert_id in enumerate(unique_experts):
                            row_for_expert[expert_id] = row_idx
                            cached = self.expert_cache.get_cached(
                                expert_id,
                                count=int(counts_list[expert_id]),
                                touch=True,
                            )
                            if cached is None:
                                cpu_rows.append(row_idx)
                                cpu_ids.append(expert_id)
                            else:
                                cached_rows.append((row_idx, cached))
                    else:
                        cpu_rows = list(range(num))
                        cpu_ids = list(unique_experts)
                        row_for_expert = {expert_id: row_idx for row_idx, expert_id in enumerate(unique_experts)}

                if cpu_rows:
                    self._pack_cpu(cpu_ids, rows=cpu_rows)

                def _copy_packed():
                    for row_idx, weights in cached_rows:
                        self._pack_gpu_from_weights(row_idx, weights)
                    if cpu_rows:
                        rows_t = torch.tensor(cpu_rows, device=self._pack_gpu_w.device, dtype=torch.int64)
                        src_w = self._pack_cpu_w[cpu_rows].to(self._pack_gpu_w.device, non_blocking=True)
                        src_s = self._pack_cpu_s[cpu_rows].to(self._pack_gpu_s.device, non_blocking=True)
                        self._pack_gpu_w.index_copy_(0, rows_t, src_w)
                        self._pack_gpu_s.index_copy_(0, rows_t, src_s)

                if self._prefetch_stream is not None:
                    with torch.cuda.stream(self._prefetch_stream):
                        _copy_packed()
                else:
                    _copy_packed()

                if self.expert_cache is not None:
                    if self.pack_cache and cpu_rows:
                        for row_idx, expert_id in zip(cpu_rows, cpu_ids):
                            weights = self._unpack_gpu(row_idx)
                            cloned = self._clone_weights(weights)
                            self.expert_cache.insert(
                                expert_id,
                                cloned,
                                count=int(counts_list[expert_id]),
                                touch=True,
                            )
                    elif not self.pack_cache and cpu_rows:
                        self.expert_cache.misses += len(cpu_rows)
            else:
                temp_weights: dict[int, tuple[torch.Tensor, ...]] = {}
                events: dict[int, torch.cuda.Event] = {}
                cached_weights: dict[int, tuple[torch.Tensor, ...]] = {}
                miss_ids: list[int] = []
                if self.expert_cache is not None:
                    for expert_id in unique_experts:
                        cached = self.expert_cache.get_cached(
                            expert_id,
                            count=int(counts_list[expert_id]),
                            touch=True,
                        )
                        if cached is None:
                            miss_ids.append(expert_id)
                        else:
                            cached_weights[expert_id] = cached
                else:
                    miss_ids = list(unique_experts)

                miss_queue = miss_ids
                next_prefetch = 0
                prefetch_window = self.prefetch_window if self._prefetch_stream is not None else 0

                def _schedule_prefetch(expert_id: int):
                    if expert_id in temp_weights:
                        return
                    if self._prefetch_stream is None:
                        expert = self.experts[expert_id]
                        temp_weights[expert_id] = expert.to_gpu(hidden_states.device)
                        return
                    with torch.cuda.stream(self._prefetch_stream):
                        expert = self.experts[expert_id]
                        weights = expert.to_gpu(hidden_states.device)
                        temp_weights[expert_id] = weights
                        evt = torch.cuda.Event()
                        evt.record(self._prefetch_stream)
                        events[expert_id] = evt

                if prefetch_window > 0:
                    for _ in range(min(prefetch_window, len(miss_queue))):
                        _schedule_prefetch(miss_queue[next_prefetch])
                        next_prefetch += 1
                else:
                    for expert_id in miss_queue:
                        _schedule_prefetch(expert_id)
            if do_profile:
                torch.cuda.synchronize()
                t_prefetch = (time.perf_counter() - t0) * 1000.0
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            if use_pack_gather:
                if self._prefetch_stream is not None:
                    torch.cuda.current_stream().wait_stream(self._prefetch_stream)
                g_w_shape, u_w_shape, d_w_shape, g_s_shape, u_s_shape, d_s_shape = self._pack_shapes
                w0, w1, w2, w3 = self._pack_w_offsets
                s0, s1, s2, _ = self._pack_s_offsets
                s2_off = self._pack_s2_offset
                pack_w = self._pack_gpu_w
                pack_s = self._pack_gpu_s
                g_w = pack_w[:, w0:w1].view(self.pack_capacity, *g_w_shape)
                u_w = pack_w[:, w1:w2].view(self.pack_capacity, *u_w_shape)
                d_w = pack_w[:, w2:w3].view(self.pack_capacity, *d_w_shape)
                g_s = pack_s[:, s0:s1].view(self.pack_capacity, *g_s_shape)
                u_s = pack_s[:, s1:s2].view(self.pack_capacity, *u_s_shape)
                d_s = pack_s[:, s2:s2_off].view(self.pack_capacity, *d_s_shape)
                g_s2 = pack_s[:, s2_off]
                u_s2 = pack_s[:, s2_off + 1]
                d_s2 = pack_s[:, s2_off + 2]

                if len(row_for_expert) != len(unique_experts):
                    raise RuntimeError("Packed MoE mapping incomplete.")
                map_cpu = torch.full((self.num_experts,), -1, dtype=torch.int32)
                for expert_id, row_idx in row_for_expert.items():
                    map_cpu[expert_id] = row_idx
                map_gpu = map_cpu.to(hidden_states.device, non_blocking=True)
                expert_ids_local = map_gpu[selected_experts]
                expert_ids_flat = expert_ids_local.reshape(-1).to(torch.int32)
                token_ids = torch.arange(hidden_states.shape[0], device=hidden_states.device).repeat_interleave(self.top_k)
                x_rep = hidden_states.index_select(0, token_ids)
                gate = nvfp4_linear_gather(x_rep, g_w, g_s, g_s2, expert_ids_flat, self.group_size)
                up = nvfp4_linear_gather(x_rep, u_w, u_s, u_s2, expert_ids_flat, self.group_size)
                act = F.silu(gate) * up
                down = nvfp4_linear_gather(act, d_w, d_s, d_s2, expert_ids_flat, self.group_size)
                weighted = down * routing_weights.reshape(-1, 1)
                final_hidden_states.index_add_(0, token_ids, weighted)
            else:
                for expert_id in unique_experts:
                    mask = selected_experts == expert_id
                    token_idx, slot_idx = torch.where(mask)
                    if token_idx.numel() == 0:
                        continue
                    if expert_id in cached_weights:
                        weights = cached_weights[expert_id]
                    else:
                        if expert_id not in temp_weights:
                            _schedule_prefetch(expert_id)
                        if expert_id in events:
                            torch.cuda.current_stream().wait_event(events[expert_id])
                        weights = temp_weights[expert_id]
                    current_state = hidden_states[token_idx]
                    current_hidden_states = self._expert_forward(current_state, weights)
                    current_hidden_states = current_hidden_states * routing_weights[token_idx, slot_idx][:, None]
                    final_hidden_states.index_add_(0, token_idx, current_hidden_states)
                    if expert_id in temp_weights and self.expert_cache is not None:
                        self.expert_cache.insert(
                            expert_id,
                            weights,
                            count=int(counts_list[expert_id]),
                            touch=True,
                        )
                    if prefetch_window > 0 and next_prefetch < len(miss_queue):
                        _schedule_prefetch(miss_queue[next_prefetch])
                        next_prefetch += 1
            if do_profile:
                torch.cuda.synchronize()
                t_expert = (time.perf_counter() - t0) * 1000.0
                hits, misses, evictions = self.expert_cache.stats()
                last_hits, last_misses, last_evictions = self._last_cache_stats
                dh, dm, de = hits - last_hits, misses - last_misses, evictions - last_evictions
                self._last_cache_stats = (hits, misses, evictions)
                moved_mb = (dm * self._expert_bytes) / (1024 ** 2)
                top_ids = unique_experts[:5]
                top_counts = [(i, counts_list[i]) for i in top_ids]
                print(
                    f"[moe-profile] layer={self.layer_idx} route_ms={t_route:.2f} prefetch_ms={t_prefetch:.2f} "
                    f"expert_ms={t_expert:.2f} unique={len(unique_experts)} hits={dh} misses={dm} evict={de} "
                    f"est_h2d_mb={moved_mb:.1f} top5_counts={top_counts}"
                )
        else:
            if do_profile:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            for k in range(self.top_k):
                expert_ids = selected_experts[:, k].to(torch.int32)
                gate = nvfp4_linear_gather(
                    hidden_states,
                    self._gate_weight,
                    self._gate_scale,
                    self._gate_scale_2,
                    expert_ids,
                    self.group_size,
                )
                up = nvfp4_linear_gather(
                    hidden_states,
                    self._up_weight,
                    self._up_scale,
                    self._up_scale_2,
                    expert_ids,
                    self.group_size,
                )
                act = F.silu(gate) * up
                out = nvfp4_linear_gather(
                    act,
                    self._down_weight,
                    self._down_scale,
                    self._down_scale_2,
                    expert_ids,
                    self.group_size,
                )
                final_hidden_states += out * routing_weights[:, k][:, None]
            if do_profile:
                torch.cuda.synchronize()
                t_expert = (time.perf_counter() - t0) * 1000.0
                print(
                    f"[moe-profile] layer={self.layer_idx} route_ms={t_route:.2f} expert_ms={t_expert:.2f}"
                )

        if len(orig_shape) > 2:
            final_hidden_states = final_hidden_states.view(*orig_shape)
        return final_hidden_states, router_logits


class Qwen3NextSparseMoeBlock(Qwen3MoeSparseMoeBlock):

    def __init__(self, config, layer_idx: int | None = None):
        super().__init__(config, layer_idx=layer_idx)
        shared_intermediate = int(getattr(config, "shared_expert_intermediate_size", 0))
        if shared_intermediate <= 0:
            raise RuntimeError("Qwen3NextSparseMoeBlock requires shared_expert_intermediate_size.")
        self.shared_expert = Qwen3NextSharedExpert(config.hidden_size, shared_intermediate)
        self.shared_expert_gate = FP16Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        moe_out, router_logits = super().forward(hidden_states)
        orig_shape = hidden_states.shape
        if hidden_states.dim() > 2:
            hidden_dim = hidden_states.shape[-1]
            flat = hidden_states.view(-1, hidden_dim)
        else:
            flat = hidden_states
        if flat.dtype != self.shared_expert_gate.weight.dtype:
            flat = flat.to(self.shared_expert_gate.weight.dtype)
        shared = self.shared_expert(flat)
        gate = torch.sigmoid(self.shared_expert_gate(flat))
        shared = shared * gate
        if hidden_states.dim() > 2:
            shared = shared.view(*orig_shape)
        moe_out = moe_out + shared
        return moe_out, router_logits


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3MoeDecoderLayer(nn.Module):

    def __init__(
        self,
        config,
        layer_idx: int,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        if tp_size != 1:
            raise RuntimeError("MoE runtime currently supports tensor_parallel_size=1.")
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        use_moe = (
            layer_idx not in getattr(config, "mlp_only_layers", [])
            and config.num_experts > 0
            and (layer_idx + 1) % config.decoder_sparse_step == 0
        )
        if use_moe:
            self.mlp = Qwen3MoeSparseMoeBlock(config, layer_idx=layer_idx)
        else:
            self.mlp = Qwen3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3MoeModel(nn.Module):

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3MoeDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3NextDecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3NextConfig,
        layer_idx: int,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        if tp_size != 1:
            raise RuntimeError("Qwen3-Next runtime currently supports tensor_parallel_size=1.")
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3NextGatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(config)
        else:
            raise ValueError(f"Unknown layer type: {self.layer_type}")

        use_moe = (
            layer_idx not in getattr(config, "mlp_only_layers", [])
            and config.num_experts > 0
            and (layer_idx + 1) % config.decoder_sparse_step == 0
        )
        if use_moe:
            self.mlp = Qwen3NextSparseMoeBlock(config, layer_idx=layer_idx)
        else:
            self.mlp = Qwen3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.dtype != self.input_layernorm.weight.dtype:
            hidden_states = hidden_states.to(self.input_layernorm.weight.dtype)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)
        else:
            hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
        hidden_states = hidden_states + residual
        residual = hidden_states
        return hidden_states, residual


class Qwen3NextModel(nn.Module):

    def __init__(
        self,
        config: Qwen3NextConfig,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3NextDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen3NextForCausalLM(nn.Module):

    def __init__(
        self,
        config: Qwen3NextConfig,
    ) -> None:
        super().__init__()
        self.model = Qwen3NextModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)


class Qwen3MoeForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.model = Qwen3MoeModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
