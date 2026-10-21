"""Qwen3.5 (Qwen3.8-27B family) architecture for AdaLLM.

Hybrid linear-attention (gated delta-net) + full-attention decoder with dense MLP.
Targets the QUASAR-QAT NVFP4 checkpoint layout:
  - split projections: linear_attn.in_proj_qkv / in_proj_z / in_proj_b / in_proj_a
  - quant tensors: <name>.weight_packed (uint8), .weight_scale (fp8 e4m3),
    .weight_global_scale (scalar), .input_global_scale (scalar, unused in W4A16 mode)
Text-only serving: interleaved mrope collapses to standard RoPE when all three
position sections are identical (no vision input), so a plain partial RoPE is used.
"""

import torch
from torch import nn
import torch.nn.functional as F

from adallm.architectures.qwen3 import (
    Qwen3NextRMSNormGated,
    Qwen3NextRMSNorm,
    _apply_rotary_pos_emb_next,
    _recurrent_gated_delta_rule,
    _HAS_VLLM_GDN,
)
from adallm.architectures.qwen3 import Qwen3MLP
from adallm.ops.attention import Attention
from adallm.ops.layernorm import RMSNorm, GemmaRMSNorm
from adallm.ops.linear import RowParallelLinear, ReplicatedLinear
from adallm.infra.context import get_context

try:
    from vllm.model_executor.layers.fla.ops import (
        fused_recurrent_gated_delta_rule,
        fused_recurrent_gated_delta_rule_packed_decode,
    )
except Exception:
    fused_recurrent_gated_delta_rule = None
    fused_recurrent_gated_delta_rule_packed_decode = None


class Qwen3_5RotaryEmbedding(nn.Module):
    """Partial RoPE for Qwen3.5 text. theta=10M, partial_rotary_factor=0.25, head_dim=256 -> rotary_dim 64."""

    def __init__(self, config, max_position: int | None = None):
        super().__init__()
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        rope_params = getattr(config, "rope_parameters", None) or {}
        partial = rope_params.get("partial_rotary_factor", getattr(config, "partial_rotary_factor", 1.0))
        base = rope_params.get("rope_theta", getattr(config, "rope_theta", 10000000))
        rotary_dim = int(head_dim * partial)
        rotary_dim = max(2, rotary_dim - (rotary_dim % 2))
        if max_position is None:
            max_position = config.max_position_embeddings
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def forward(self, positions: torch.Tensor):
        if positions.dim() == 0:
            positions = positions.view(1)
        elif positions.dim() > 1:
            positions = positions.reshape(-1)
        cos = self.cos_cache.index_select(0, positions)
        sin = self.sin_cache.index_select(0, positions)
        return cos, sin


class Qwen3_5GatedDeltaNet(nn.Module):
    """Gated delta-net with QUASAR split projections (in_proj_qkv/z/b/a)."""

    def __init__(self, config, layer_idx: int):
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

        # Split projections matching checkpoint names exactly.
        self.in_proj_qkv = ReplicatedLinear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = ReplicatedLinear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = ReplicatedLinear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = ReplicatedLinear(self.hidden_size, self.num_v_heads, bias=False)

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.norm = Qwen3NextRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.out_proj = RowParallelLinear(self.value_dim, self.hidden_size, bias=False)

        # CUDA-graph-safe recurrent state pool (vLLM mamba-style):
        #   mamba_state: [num_slots, HV, V, K]; conv_state: [num_slots, conv_dim, kernel-1]
        # Allocated/zeroed by ModelRunner.allocate_kv_cache(); indexed via
        # context.delta_slots (a GPU int64 tensor), so state updates happen in
        # kernels and survive graph replay. Slot 0 is reserved as the always-
        # zero NULL slot (kernels skip state_idx <= 0 → outputs zeros).
        self.mamba_state: torch.Tensor | None = None
        self.conv_state: torch.Tensor | None = None
        self.num_state_slots = 0

    def allocate_state_pool(self, num_slots: int, device, dtype):
        assert num_slots >= 1
        self.num_state_slots = num_slots
        self.mamba_state = torch.zeros(
            num_slots, self.num_v_heads, self.head_v_dim, self.head_k_dim,
            device=device, dtype=dtype,
        )
        self.conv_state = torch.zeros(
            num_slots, self.conv_dim, self.conv_kernel_size - 1,
            device=device, dtype=dtype,
        )

    def _split_qkv(self, mixed_qkv: torch.Tensor):
        # HF qwen3_5 layout is contiguous: [Q(key_dim) | K(key_dim) | V(value_dim)]
        # (unlike Qwen3-Next's fused per-group-interleaved qkvz ordering).
        return torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_context()
        input_dtype = hidden_states.dtype
        if self.mamba_state is None or self.conv_state is None:
            raise RuntimeError("delta state pool not allocated (call allocate_state_pool)")

        mixed_qkv = self.in_proj_qkv(hidden_states)   # [T, key_dim*2 + value_dim]
        z = self.in_proj_z(hidden_states)             # [T, value_dim]
        b = self.in_proj_b(hidden_states)             # [T, num_v_heads]
        a = self.in_proj_a(hidden_states)             # [T, num_v_heads]

        if context.is_prefill:
            # ---- varlen prefill (chunked): cu_seqlens_q is chunk-relative ----
            cu = context.cu_seqlens_q
            n_seqs = cu.shape[0] - 1
            slots = context.delta_slots              # [n_seqs] int64
            conv_out = torch.empty_like(mixed_qkv)
            weight = self.conv1d.weight.squeeze(1)   # [conv_dim, kernel]
            kml1 = self.conv_kernel_size - 1
            for i in range(n_seqs):
                s0, s1 = int(cu[i]), int(cu[i + 1])
                slot = int(slots[i])
                x = mixed_qkv[s0:s1].transpose(0, 1)                 # [conv_dim, T]
                prev = self.conv_state[slot]                         # zeros for fresh seqs
                window = torch.cat([prev, x], dim=-1)                # [conv_dim, T+k-1]
                out = F.conv1d(window, self.conv1d.weight, groups=self.conv_dim, padding=0)
                conv_out[s0:s1] = F.silu(out).transpose(0, 1)
                # carry last k-1 columns (covers x, possibly zeros) into pool
                self.conv_state[slot] = window[:, -kml1:].detach()

            total = conv_out.shape[0]
            q = conv_out[:, : self.key_dim].view(1, total, self.num_k_heads, self.head_k_dim)
            k = conv_out[:, self.key_dim : 2 * self.key_dim].view(1, total, self.num_k_heads, self.head_k_dim)
            v = conv_out[:, 2 * self.key_dim :].view(1, total, self.num_v_heads, self.head_v_dim)
            g = (-self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)).view(1, total, self.num_v_heads)
            beta = torch.sigmoid(b).view(1, total, self.num_v_heads)

            # Kernel reads ssm_state_indices[i_n, i_t] per token → needs a 2-D
            # [N, T] tensor (1-D would read OOB at i_t > 0 → illegal access).
            lens = [int(cu[i + 1] - cu[i]) for i in range(n_seqs)]
            max_len = max(lens) if lens else 1
            idx2d = slots.new_zeros(n_seqs, max(max_len, 1))
            for i in range(n_seqs):
                idx2d[i, : lens[i]] = slots[i]

            core_attn_out, _ = fused_recurrent_gated_delta_rule(
                q=q.contiguous(), k=k.contiguous(), v=v.contiguous(),
                g=g.contiguous(), beta=beta.contiguous(),
                initial_state=self.mamba_state,
                inplace_final_state=True,
                cu_seqlens=cu,
                ssm_state_indices=idx2d,
                use_qk_l2norm_in_kernel=True,
            )
            core_attn_out = core_attn_out.squeeze(0)                 # [T, HV*V]
        else:
            # ---- decode: batch of single tokens, state pool addressing ----
            B = hidden_states.shape[0]
            slots = context.delta_slots                              # [B] int64
            state = self.conv_state.index_select(0, slots)           # [B, conv_dim, k-1]
            window = torch.cat(
                [state.transpose(1, 2), mixed_qkv.unsqueeze(1)], dim=1
            )                                                        # [B, kernel, conv_dim]
            w = self.conv1d.weight.squeeze(1)                        # [conv_dim, kernel]
            conv_out = F.silu((window * w.t().unsqueeze(0)).sum(dim=1))  # [B, conv_dim]
            self.conv_state.index_copy_(0, slots, window[:, 1:, :].transpose(1, 2).contiguous())

            out_buf = torch.empty(
                B, 1, self.num_v_heads, self.head_v_dim,
                device=mixed_qkv.device, dtype=mixed_qkv.dtype,
            )
            fused_recurrent_gated_delta_rule_packed_decode(
                mixed_qkv=conv_out.contiguous(),
                a=a.contiguous(),
                b=b.contiguous(),
                A_log=self.A_log.contiguous(),
                dt_bias=self.dt_bias.contiguous(),
                scale=self.head_k_dim ** -0.5,
                initial_state=self.mamba_state,
                out=out_buf,
                ssm_state_indices=slots,
                use_qk_l2norm_in_kernel=True,
            )
            core_attn_out = out_buf.squeeze(1)                       # [B, HV*V]

        # Norm applies per value head over head_v_dim (matching Qwen3-Next /
        # HF qwen3_5 granularity).
        b_remain = core_attn_out.shape[0]
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(b_remain, self.value_dim)

        output = self.out_proj(core_attn_out)
        if output.dtype != input_dtype:
            output = output.to(input_dtype)
        return output


class Qwen3_5Attention(nn.Module):
    """Gated full attention: q_proj emits 2x head_dim (q + sigmoid gate)."""

    def __init__(self, config, max_position: int | None = None):
        super().__init__()
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = ReplicatedLinear(config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=False)
        self.k_proj = ReplicatedLinear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = ReplicatedLinear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = RowParallelLinear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        # qwen3_5 norms are zero-centered: y = x_hat * (1 + w)
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        if max_position is None:
            max_position = config.max_position_embeddings
        self.rotary_emb = Qwen3_5RotaryEmbedding(config, max_position=max_position)
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
        if cos.dim() == 3 and cos.shape[0] == 1:
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


class Qwen3_5MLP(nn.Module):
    """Separate gate/up/down projections (NOT fused): QUASAR quantizes each
    with its own per-tensor global scale, which a merged gate_up_proj's single
    weight_scale_2 cannot represent."""

    def __init__(self, config):
        super().__init__()
        from adallm.ops.linear import ReplicatedLinear, RowParallelLinear
        from adallm.ops.activation import SiluAndMul
        self.gate_proj = ReplicatedLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = ReplicatedLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = RowParallelLinear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(self.act_fn(torch.cat([gate, up], dim=-1)))


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int, max_position: int | None = None):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, max_position=max_position)
        else:
            raise ValueError(f"Unknown layer type: {self.layer_type}")
        self.mlp = Qwen3_5MLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
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
        return hidden_states, residual


class Qwen3_5TextModel(nn.Module):
    def __init__(self, config, max_position: int | None = None):
        super().__init__()
        from adallm.ops.embed_head import VocabParallelEmbedding

        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, i, max_position=max_position) for i in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen3_5Model(nn.Module):
    """Wrapper matching checkpoint prefix `model.language_model.*` (vision ignored)."""

    def __init__(self, config, max_position: int | None = None):
        super().__init__()
        self.language_model = Qwen3_5TextModel(config, max_position=max_position)

    def forward(self, input_ids, positions):
        return self.language_model(input_ids, positions)


class Qwen3_5ForCausalLM(nn.Module):
    # Separate gate/up projections in this architecture (see Qwen3_5MLP):
    # QUASAR per-tensor scales require unfused modules. No packed mapping.

    def __init__(self, config, max_position: int | None = None):
        super().__init__()
        self.model = Qwen3_5Model(config, max_position=max_position)
        from adallm.ops.embed_head import ParallelLMHead

        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight.data = self.model.language_model.embed_tokens.weight.data

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)
