"""Qwen4Exp (Qwen3.8-Flash-Next) architecture for AdaLLM.

Model: nvidia/Qwen3.8-Flash-Next-NVFP4 (model_type `qwen4_exp`, arch
Qwen4ExpForConditionalGeneration). Text-only serving; vision tower ignored.

Layout summary vs existing paths:
- GatedDeltaNet layers: IDENTICAL tensor layout to qwen3_5 (in_proj_qkv /
  in_proj_z / in_proj_b / in_proj_a split, sigmoid output gate, Gemma
  zero-centered norms) -> subclass Qwen3_5GatedDeltaNet (keeps the Round8-E
  TP=2 head sharding and the graph-safe state pools).
- Full-attention layers: gated attention, q emits 2x head_dim (q + sigmoid
  gate), GQA 2 kv heads, Gemma q/k norms, partial RoPE 64/256 theta 10M ->
  reuse Qwen3_5Attention. QSA ngram indexer weights are loaded but the
  indexer forward is approximated: indexer_budget=2048 with compress_ratio 4
  means top-512-of-4-token-blocks = 2048 selected tokens, so for contexts
  <= 2048 EVERY block is selected and full causal attention is EXACT. For
  longer contexts we fall back to a sliding window of 2048 tokens (recent
  tokens) instead of learned retrieval - a documented quality approximation
  (real block top-k retrieval is a stretch goal, not implemented).
- Hyper-connections (GatedResidual): 4 residual streams, per-layer
  attn/mlp hyper_connection modules with grouped (per-stream) GemmaRMSNorm,
  low-rank (320) sigmoid input mixing and 2*sigmoid injection weights. A
  final hyper_connection_mixer (no injection) folds the streams back to
  [T, hidden] - this REPLACES the usual final RMSNorm.
- PLE (Per-Layer Embedding, layer_idx 1 only): hashed trigram embedding
  table (16 heads x ~20M prime vocab x 160 dims FP8 = ~51GB) + key/value
  projections + dilated depthwise conv. The 51GB table lives in the
  safetensors FILE, accessed via np.memmap (read-only, shared page cache
  between both TP ranks, zero RAM/memlock cost - the box caps MEMLOCK at
  12GB so pinning is impossible). The ngram hash + row gather run on the
  CPU from host input-id mirrors in the context; rows are dequantized via
  a 256-entry LUT (fp8 * scalar scale) and pushed through a small pinned
  staging buffer. This keeps the decode step host-sync-free (no GPU->CPU
  copies) at the cost of ~1ms CPU per prefill chunk.
- QSA indexer: budget 2048 -> for contexts <= 2048 every 4-token block is
  selected, so FULL causal attention is EXACT. We approximate longer
  contexts with full attention too (dense superset of the top-2048
  retrieval - closer to reference quality than a recent-token window).
  Indexer weights are loaded but unused.
- MoE: 512 experts x moe_intermediate 640, top-10, norm_topk_prob, shared
  expert (640) with sigmoid gate. Experts are NVFP4-packed and GPU-resident
  EXPERT-PARALLEL across TP ranks (contiguous halves, 256/rank at TP=2;
  replication would need 75GB/GPU - impossible on 48GB): each rank computes
  the partial sum over its local experts for ALL tokens (hidden states are
  replicated in this TP scheme) and the partials are all-reduced. The
  shared expert is added on rank 0 before the reduce.

Checkpoint quant (ModelOpt MIXED_PRECISION): experts NVFP4 gs=16 (uint8
packed + fp8 per-group scales + fp32 scale_2), everything else BF16
(including the FP8 PLE table with scalar weight_scale ~2e-4). Dense weights
stay BF16 (FP16Linear); experts feed the existing nvfp4_linear_gather
Triton kernel.
"""

import json
import math
import re

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from adallm.architectures.qwen3_5 import (
    Qwen3_5GatedDeltaNet,
    Qwen3_5Attention,
)
from adallm.ops.linear import FP16Linear
from adallm.infra.context import get_context


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------

class GroupedGemmaRMSNorm(nn.Module):
    """Zero-centered Gemma RMSNorm applied per group: x_hat * (1 + w).

    weight length dim; dim must be divisible by group_size (None = plain)."""

    def __init__(self, dim: int, group_size: int | None, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.group_size = group_size
        if group_size is not None and dim % group_size != 0:
            raise ValueError(f"{dim} not divisible by {group_size}")
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.dtype
        xf = x.float()
        if self.group_size is not None:
            xf = xf.reshape(*xf.shape[:-1], -1, self.group_size)
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.group_size is not None:
            xf = xf.flatten(-2)
        xf = xf * (1.0 + self.weight.float())
        return xf.to(orig)


# ---------------------------------------------------------------------------
# Hyper-connection (GatedResidual)
# ---------------------------------------------------------------------------

class Qwen4ExpGatedResidual(nn.Module):
    """Gated hyper-connection. Module attribute names match the checkpoint:

      hc_norm.weight                 [hc_count*hidden]
      input_mix_weight_down.weight   [hc_lowrank, hc_count*hidden]
      input_mix_weight_up.weight     [hc_count*hidden, hc_lowrank]
      block_inject_weight.weight     [hc_count, hc_count*hidden]  (combine only)
    """

    def __init__(self, config, use_combine: bool = True):
        super().__init__()
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.hc_hidden = self.hc_count * self.hidden_size
        self.hc_lowrank = config.hc_lowrank
        self.use_combine = use_combine
        self.hc_norm = GroupedGemmaRMSNorm(self.hc_hidden, self.hidden_size, eps=config.rms_norm_eps)
        self.input_mix_weight_down = FP16Linear(self.hc_hidden, self.hc_lowrank, bias=False)
        self.input_mix_weight_up = FP16Linear(self.hc_lowrank, self.hc_hidden, bias=False)
        if use_combine:
            self.block_inject_weight = FP16Linear(self.hc_hidden, self.hc_count, bias=False)

    def mix(self, hyper_input: torch.Tensor):
        """Returns (block_input [T,H], hyper_input, injection [T,hc_count] or None)."""
        xn = self.hc_norm(hyper_input)
        lowrank = F.silu(self.input_mix_weight_down(xn) / self.hc_count)
        gate = torch.sigmoid(self.input_mix_weight_up(lowrank))
        gate = gate.unflatten(-1, (self.hc_count, self.hidden_size))
        xn_s = xn.unflatten(-1, (self.hc_count, self.hidden_size))
        block_input = (gate * xn_s).mean(dim=-2)
        if not self.use_combine:
            return block_input, hyper_input, None
        injection = 2.0 * torch.sigmoid(self.block_inject_weight(xn) / self.hc_count)
        return block_input, hyper_input, injection

    @staticmethod
    def combine(hyper_input, block_output, injection):
        """hyper + block_output * injection -> flattened [T, hc*H]."""
        summed = block_output.unsqueeze(-2) * injection.unsqueeze(-1)
        return hyper_input + summed.flatten(-2)


# ---------------------------------------------------------------------------
# PLE (per-layer embedding)
# ---------------------------------------------------------------------------

def _is_prime(v: int) -> bool:
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    for d in range(3, math.isqrt(v) + 1, 2):
        if v % d == 0:
            return False
    return True


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _build_layer_multipliers(unigram_vocab_size, ngram_size, ple_layer_index, seed: int):
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    multipliers = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
    return torch.tensor(multipliers, dtype=torch.int64)


class Qwen4ExpNgramEmbedding(nn.Module):
    """Hashed trigram embedding table (~320M rows x 160 dims FP8, ~51GB).

    The table is NEVER resident in RAM: rows are gathered straight from the
    safetensors file via np.memmap (read-only, shared page cache across TP
    ranks). Hashing runs host-side on token-id mirrors (reference math:
    splitmix multipliers, per-ngram xor mix, eos-segment-aware shifts);
    gathered uint8 rows go through a pinned staging buffer, are dequantized
    on GPU (fp8 view * scalar scale) and land in a persistent GPU stage
    buffer that the PLE layer reads - so decode stays CUDA-graph-safe with
    zero device->host syncs.
    """

    def __init__(self, config, embedding_dim: int, layer_idx: int, ple_layer_index: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.ngram_size = config.ngram_size
        self.context_len = self.ngram_size - 1
        self.heads_per_ngram = config.heads_per_ngram
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.ple_layer_index = ple_layer_index
        self.unigram_vocab_size = config.vocab_size
        self.ngram_vocab_size_base = config.ngram_vocab_size_base
        self.seed = config.seed
        eos = config.eos_token_id
        self.eos_token_id = eos[0] if isinstance(eos, (list, tuple)) else eos

        head_dim_per_ngram = embedding_dim // self.ngram_heads
        self.head_dim = head_dim_per_ngram
        self.embedding_dim = embedding_dim
        self.head_vocab_sizes = []
        self.head_offsets = []
        total = 0
        for head_idx in range(self.ngram_heads):
            global_head_idx = self.ple_layer_index * self.ngram_heads + head_idx
            size = _find_nth_prime_after(self.ngram_vocab_size_base - 1, global_head_idx + 1)
            self.head_vocab_sizes.append(size)
            self.head_offsets.append(total)
            total += size
        divisor = config.make_ngram_vocab_size_divisible_by
        self.total_vocab_size = math.ceil(total / divisor) * divisor

        self.register_buffer("layer_multipliers", _build_layer_multipliers(
            self.unigram_vocab_size, self.ngram_size, self.ple_layer_index, self.seed))
        self.register_buffer("ngram_heads_vocab_sizes",
                             torch.tensor(self.head_vocab_sizes, dtype=torch.long))
        self.register_buffer("ngram_heads_offsets",
                             torch.tensor(self.head_offsets, dtype=torch.long))

        self.weight_scale = 1.0            # loaded from checkpoint scalar
        self._file = None                  # np.memmap per shard (lazy)
        self._shard_offsets: list[int] | None = None   # byte offset per shard
        self._rows_per_shard = 0
        self._num_shards = 0
        try:
            import numpy as _numpy
            self._np = _numpy
        except ImportError:
            self._np = None

    # -- lazy mmap attach ----------------------------------------------------
    def attach_file(self, path: str):
        if self._file is not None:
            return
        import numpy as np
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(n))
        base = 8 + n
        pref = f"model.language_model.layers.{self.layer_idx}.ple.ple_embedding.ngram_embedding.shard_"
        shards = {}
        for k, v in header.items():
            if k.startswith(pref) and k.endswith(".weight"):
                sid = int(k[len(pref):-len(".weight")])
                shards[sid] = v["data_offsets"]
        self._num_shards = len(shards)
        self._rows_per_shard = (shards[0][1] - shards[0][0]) // self.head_dim
        self._shard_offsets = [base + shards[i][0] for i in range(self._num_shards)]
        self._path = path
        self._np = np

    def _gather_rows(self, row_ids) -> "np.ndarray":
        """row_ids: np.int64 [N] global rows -> uint8 [N, head_dim]."""
        np_ = self._np
        sid = row_ids // self._rows_per_shard
        local = row_ids % self._rows_per_shard
        out = np_.empty((row_ids.shape[0], self.head_dim), dtype=np_.uint8)
        for s in np_.unique(sid):
            m = sid == s
            mm = np_.memmap(self._path, dtype=np_.uint8, mode="r",
                            offset=self._shard_offsets[s],
                            shape=(self._rows_per_shard, self.head_dim))
            out[m] = mm[local[m]]
        return out

    # -- host-side hashing (reference math, numpy) ----------------------------
    def _shift_right_ignore_eos(self, hist, shift: int, eos: int):
        np_ = self._np
        if shift == 0:
            return hist
        L = hist.shape[0]
        pos = np_.arange(L)
        eos_pos = np_.where(hist == eos, pos, -1)
        prev_inc = np_.maximum.accumulate(eos_pos)
        seg_start = np_.concatenate([[0], prev_inc[:-1] + 1])
        pos_in_seg = pos - seg_start
        src = np_.clip(pos - shift, 0, None)
        shifted = hist[src]
        valid = (pos_in_seg >= shift) & (pos - shift >= 0)
        return np_.where(valid, shifted, eos)

    def host_rows(self, hist, n_out: int):
        """hist [L] int64 (eos-padded context + tokens) -> rows for the last
        n_out tokens: [n_out, ngram_heads] global row ids."""
        rows = self.host_ngram_rows(hist)
        return rows[-n_out:]

    def host_ngram_rows(self, hist) -> "np.ndarray":
        """hist [L] token ids (numpy int64) -> [L, ngram_heads] global rows."""
        np_ = self._np
        mult = self.layer_multipliers.numpy()
        eos = self.eos_token_id
        shifted = [self._shift_right_ignore_eos(hist, s, eos) for s in range(self.ngram_size)]
        mixed = shifted[0] * mult[0]
        for p in range(1, self.ngram_size):
            mixed = np_.bitwise_xor(mixed, shifted[p] * mult[p])
        hpg = self.heads_per_ngram
        vs = self.ngram_heads_vocab_sizes.numpy()
        off = self.ngram_heads_offsets.numpy()
        cols = []
        for ngram in range(2, self.ngram_size + 1):
            h0 = (ngram - 2) * hpg
            ids_h = np_.remainder(mixed[:, None], vs[None, h0:h0 + hpg]) + off[None, h0:h0 + hpg]
            cols.append(ids_h)
        return np_.concatenate(cols, axis=1)                     # [L, heads]

    # -- per-step host prepare (called by ModelRunner before forward) ---------
    def prepare_step(self, segments, stage_gpu: torch.Tensor, scratch_u8: torch.Tensor):
        """segments: list of (hist_ids: list[int], chunk_ids: list[int]).
        Fills stage_gpu[0:T] with the dequantized [T, embedding_dim] PLE
        embedding concat (heads*head_dim columns)."""
        np_ = self._np
        def _ints(v):
            if torch.is_tensor(v):
                return [int(t) for t in v.detach().cpu().tolist()]
            return [int(t) for t in v]
        rows_all = [self.host_rows(np_.asarray(_ints(h), dtype=np_.int64), len(c))
                    for (h, c) in segments]
        rows = np_.concatenate(rows_all, axis=0) if len(rows_all) > 1 else rows_all[0]
        T = rows.shape[0]
        flat = np_.ascontiguousarray(rows.reshape(-1))
        gathered = self._gather_rows(flat)                     # [T*heads, dim] u8
        host_t = torch.from_numpy(gathered)
        n = host_t.numel()
        scratch_u8[:n].copy_(host_t, non_blocking=True)
        g = scratch_u8[:n].cuda(non_blocking=True).view(T, self.ngram_heads, self.head_dim)
        if self._scale_gpu is None or self._scale_gpu.device != g.device:
            self._scale_gpu = torch.tensor(float(self.weight_scale),
                                           dtype=torch.float16, device=g.device)
        emb = g.view(torch.float8_e4m3fn).to(stage_gpu.dtype) * self._scale_gpu.to(stage_gpu.dtype)
        stage_gpu[:T].view(T, self.ngram_heads, self.head_dim).copy_(emb, non_blocking=True)


class Qwen4ExpPLELayer(nn.Module):
    """Per-layer embedding injection (one layer, from ple_layer_ids).

    The ngram table read + hashing happen HOST-side per step
    (NgramEmbedding.prepare_step, driven by ModelRunner) into a persistent
    GPU stage buffer; forward() only consumes the stage buffer + a per-slot
    dilated-conv state pool (decode is CUDA-graph safe)."""

    def __init__(self, config, layer_idx: int, ple_layer_index: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        hc_hidden = self.hidden_size * self.hc_count
        self.ple_embedding = Qwen4ExpNgramEmbedding(config, config.ple_embed_dim, layer_idx, ple_layer_index)
        conv_kernel = config.ple_conv_kernel_size
        self.conv_dilation = config.ngram_size
        self.conv_kernel = conv_kernel
        self.short_conv_state_len = (conv_kernel - 1) * self.conv_dilation   # 3*3 = 9
        self.key_proj = FP16Linear(config.ple_embed_dim, hc_hidden, bias=False)
        self.value_proj = FP16Linear(config.ple_embed_dim, self.hidden_size, bias=False)
        self.norm_key = GroupedGemmaRMSNorm(hc_hidden, self.hidden_size, eps=config.rms_norm_eps)
        self.norm_query = GroupedGemmaRMSNorm(hc_hidden, self.hidden_size, eps=config.rms_norm_eps)
        self.norm_conv = GroupedGemmaRMSNorm(hc_hidden, self.hidden_size, eps=config.rms_norm_eps)
        self.conv1d = nn.Conv1d(hc_hidden, hc_hidden, kernel_size=conv_kernel,
                                groups=hc_hidden, dilation=self.conv_dilation, bias=False)
        self.ple_conv_state: torch.Tensor | None = None   # [slots, C, 9]
        self.stage: torch.Tensor | None = None            # [max_tokens, E]
        self.num_state_slots = 0

    def allocate_state_pool(self, num_slots: int, max_tokens: int, device, dtype):
        self.num_state_slots = num_slots
        C = self.hidden_size * self.hc_count
        self.ple_conv_state = torch.zeros(num_slots, C, self.short_conv_state_len,
                                          device=device, dtype=dtype)
        self.stage = torch.zeros(max_tokens, self.ple_embedding.embedding_dim,
                                 device=device, dtype=dtype)

    def reset_slot(self, slot: int):
        if self.ple_conv_state is not None:
            self.ple_conv_state[slot].zero_()

    def _ple_gated(self, hidden_states, emb):
        hc, H = self.hc_count, self.hidden_size
        key_normed = self.norm_key(self.key_proj(emb)).view(*emb.shape[:-1], hc, H)
        value = self.value_proj(emb)
        query_normed = self.norm_query(hidden_states).view(*hidden_states.shape[:-1], hc, H)
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(H)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        return torch.sigmoid(gate) * value.unsqueeze(-2)     # [..., hc, H]

    def forward(self, hidden_states: torch.Tensor, bounds, is_prefill: bool) -> torch.Tensor:
        """hidden_states [T, hc*H]; prefill bounds: host list (s0, s1, slot)."""
        T = hidden_states.shape[0]
        emb = self.stage[:T]
        gated = self._ple_gated(hidden_states, emb)          # [T, hc, H]
        gvn = self.norm_conv(gated.flatten(-2))              # [T, C]
        if is_prefill:
            outs = []
            for (s0, s1, slot) in bounds:
                state = self.ple_conv_state[slot]            # [C, 9]
                xt = gvn[s0:s1].transpose(0, 1).contiguous() # [C, L]
                window = torch.cat([state, xt], dim=-1)      # [C, 9+L]
                conv = F.conv1d(window.unsqueeze(0), self.conv1d.weight,
                                groups=window.shape[0], dilation=self.conv_dilation)
                conv_out = F.silu(conv.squeeze(0)).transpose(0, 1).contiguous()
                self.ple_conv_state[slot] = window[:, -self.short_conv_state_len:].detach()
                outs.append(gated[s0:s1].flatten(-2) + conv_out)
            return torch.cat(outs, dim=0)

        # decode: batched, graph-safe (device slot addressing)
        slots = get_context().delta_slots                    # [B] i64
        state = self.ple_conv_state.index_select(0, slots)   # [B, C, 9]
        window = torch.cat([state, gvn.unsqueeze(-1)], dim=-1)  # [B, C, 10]
        w = self.conv1d.weight.squeeze(1)                    # [C, k]
        acc = window[:, :, 0] * w[:, 0].unsqueeze(0)
        for j in range(1, self.conv_kernel):
            acc = acc + window[:, :, j * self.conv_dilation] * w[:, j].unsqueeze(0)
        conv_out = F.silu(acc)                               # [B, C]
        self.ple_conv_state.index_copy_(0, slots, window[:, :, 1:].detach())
        return gated.flatten(-2) + conv_out


# ---------------------------------------------------------------------------
# MoE (expert-parallel, GPU-resident NVFP4)
# ---------------------------------------------------------------------------

class Qwen4ExpSparseMoeBlock(nn.Module):
    """Expert-parallel NVFP4 MoE. Two placement modes:

    - GPU-resident EP (default): flat packed params [E_local, ...] on GPU.
    - Offload staging (ADALLM_Q4E_EP_OFFLOAD=1, or =partial for alternate
      layers): params live in CPU RAM (unpinned; MEMLOCK caps at 12GB); each
      forward copies the chunk(s) of UNIQUE selected local experts into a
      persistent GPU staging buffer and runs the gather kernel against it.
      Prefill may select most local experts -> chunked waves (chunk=48
      experts, ~59MB H2D each); decode typically needs a single chunk.
    """

    _STAGE_CHUNK = 48

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size
        self.moe_intermediate = config.moe_intermediate_size
        self.gate = FP16Linear(config.hidden_size, self.num_experts, bias=False)

        tp = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        self.tp_size = tp
        self.tp_rank = rank
        assert self.num_experts % tp == 0
        self.local_experts = self.num_experts // tp
        self.expert_base = rank * self.local_experts

        import os as _os
        _off = _os.getenv("ADALLM_Q4E_EP_OFFLOAD", "0").strip().lower()
        if _off == "partial":
            self.offload = (layer_idx % 2 == 1)      # half the layers offload
        else:
            self.offload = _off in ("1", "true", "yes", "on")

        from adallm.infra.quant import get_quant_config
        qcfg = get_quant_config()
        gs = qcfg.group_size if qcfg.enabled else 16
        self.group_size = gs
        E, I, Hh = self.local_experts, self.moe_intermediate, config.hidden_size

        dev = "cpu" if self.offload or not torch.cuda.is_available() else "cuda"

        def param(shape, dtype):
            return nn.Parameter(torch.empty(*shape, dtype=dtype, device=dev), requires_grad=False)

        self._gate_weight = param((E, I, Hh // 2), torch.uint8)
        self._gate_scale = param((E, I, Hh // gs), torch.float16)
        self._gate_scale_2 = param((E,), torch.float32)
        self._up_weight = param((E, I, Hh // 2), torch.uint8)
        self._up_scale = param((E, I, Hh // gs), torch.float16)
        self._up_scale_2 = param((E,), torch.float32)
        self._down_weight = param((E, Hh, I // 2), torch.uint8)
        self._down_scale = param((E, Hh, I // gs), torch.float16)
        self._down_scale_2 = param((E,), torch.float32)
        for p in (self._gate_weight, self._gate_scale, self._gate_scale_2,
                  self._up_weight, self._up_scale, self._up_scale_2,
                  self._down_weight, self._down_scale, self._down_scale_2):
            p.weight_loader = self._ep_row_loader
        self._stage = None        # (row_map tensor, staged params) per step

        shared_inter = int(getattr(config, "shared_expert_intermediate_size", 0))
        self.shared_expert_gate = FP16Linear(config.hidden_size, 1, bias=False)
        self._has_shared = shared_inter > 0
        if self._has_shared:
            self.shared_expert = nn.Module()
            self.shared_expert.gate_proj = FP16Linear(config.hidden_size, shared_inter, bias=False)
            self.shared_expert.up_proj = FP16Linear(config.hidden_size, shared_inter, bias=False)
            self.shared_expert.down_proj = FP16Linear(shared_inter, config.hidden_size, bias=False)

    def expert_is_local(self, gid: int) -> bool:
        return self.expert_base <= gid < self.expert_base + self.local_experts

    def _ep_row_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, gid: int):
        local = int(gid) - self.expert_base
        param.data[local].copy_(loaded_weight)

    def _forward_resident(self, x, orig_dtype, loc, loc_c, is_local, routing_weights):
        from adallm.ops.nvfp4_gemm import nvfp4_linear_gather
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            ids = loc_c[:, k]
            g = nvfp4_linear_gather(x, self._gate_weight.data, self._gate_scale.data,
                                    self._gate_scale_2.data, ids, self.group_size).to(orig_dtype)
            u = nvfp4_linear_gather(x, self._up_weight.data, self._up_scale.data,
                                    self._up_scale_2.data, ids, self.group_size).to(orig_dtype)
            act = F.silu(g) * u
            d = nvfp4_linear_gather(act, self._down_weight.data, self._down_scale.data,
                                    self._down_scale_2.data, ids, self.group_size).to(orig_dtype)
            w = routing_weights[:, k] * is_local[:, k].to(routing_weights.dtype)
            out = out + d * w.unsqueeze(-1)
        return out

    def _stage_chunk_gpu(self, local_ids: torch.Tensor):
        """Copy the given unique LOCAL expert rows (int64 CPU->GPU tensor) of
        all 9 flat params into persistent GPU staging tensors; returns dict of
        (staged param, row_map) where row_map maps staged row -> local id."""
        idx = local_ids  # [n] int64 on GPU
        n = idx.numel()
        st = self._stage
        if st is None or st["cap"] < n:
            cap = max(n, self._STAGE_CHUNK)
            st = {"cap": cap, "gw": torch.empty(cap, *self._gate_weight.shape[1:], dtype=torch.uint8, device="cuda"),
                  "gs": torch.empty(cap, *self._gate_scale.shape[1:], dtype=torch.float16, device="cuda"),
                  "g2": torch.empty(cap, dtype=torch.float32, device="cuda"),
                  "uw": torch.empty(cap, *self._up_weight.shape[1:], dtype=torch.uint8, device="cuda"),
                  "us": torch.empty(cap, *self._up_scale.shape[1:], dtype=torch.float16, device="cuda"),
                  "u2": torch.empty(cap, dtype=torch.float32, device="cuda"),
                  "dw": torch.empty(cap, *self._down_weight.shape[1:], dtype=torch.uint8, device="cuda"),
                  "ds": torch.empty(cap, *self._down_scale.shape[1:], dtype=torch.float16, device="cuda"),
                  "d2": torch.empty(cap, dtype=torch.float32, device="cuda")}
            self._stage = st
        st["gw"][:n].copy_(self._gate_weight.data.index_select(0, idx.cpu()).cuda(non_blocking=True))
        st["gs"][:n].copy_(self._gate_scale.data.index_select(0, idx.cpu()).cuda(non_blocking=True))
        st["g2"][:n].copy_(self._gate_scale_2.data.index_select(0, idx.cpu()).cuda(non_blocking=True))
        st["uw"][:n].copy_(self._up_weight.data.index_select(0, idx.cpu()).cuda(non_blocking=True))
        st["us"][:n].copy_(self._up_scale.data.index_select(0, idx.cpu()).cuda(non_blocking=True))
        st["u2"][:n].copy_(self._up_scale_2.data.index_select(0, idx.cpu()).cuda(non_blocking=True))
        st["dw"][:n].copy_(self._down_weight.data.index_select(0, idx.cpu()).cuda(non_blocking=True))
        st["ds"][:n].copy_(self._down_scale.data.index_select(0, idx.cpu()).cuda(non_blocking=True))
        st["d2"][:n].copy_(self._down_scale_2.data.index_select(0, idx.cpu()).cuda(non_blocking=True))
        # map local id -> staged row
        row_map = torch.full((self.local_experts,), -1, dtype=torch.int32, device="cuda")
        row_map[idx] = torch.arange(n, dtype=torch.int32, device="cuda")
        return st, row_map

    def _forward_offload(self, x, orig_dtype, loc, loc_c, is_local, routing_weights):
        from adallm.ops.nvfp4_gemm import nvfp4_linear_gather
        T, k = loc.shape
        flat = loc.reshape(-1)
        flat_valid = flat[(flat >= 0) & (flat < self.local_experts)]
        unique = torch.unique(flat_valid)                     # [n] int64 GPU
        out = torch.zeros_like(x)
        n_unique = unique.numel()
        if n_unique == 0:
            return out
        chunk = self._STAGE_CHUNK
        for c0 in range(0, n_unique, chunk):
            uc = unique[c0:c0 + chunk]
            st, row_map = self._stage_chunk_gpu(uc)
            staged_ids_all = row_map[loc_c]                   # [T, k] (-1 = not this wave/remote)
            for k_i in range(k):
                ids = staged_ids_all[:, k_i]
                rows_ok = ids >= 0
                if not bool(rows_ok.any()):
                    continue
                ids = ids.clamp_min(0)
                g = nvfp4_linear_gather(x, st["gw"], st["gs"], st["g2"], ids, self.group_size).to(orig_dtype)
                u = nvfp4_linear_gather(x, st["uw"], st["us"], st["u2"], ids, self.group_size).to(orig_dtype)
                act = F.silu(g) * u
                d = nvfp4_linear_gather(act, st["dw"], st["ds"], st["d2"], ids, self.group_size).to(orig_dtype)
                w = routing_weights[:, k_i] * rows_ok.to(routing_weights.dtype)
                out = out + d * w.unsqueeze(-1)
        return out

    def forward(self, hidden_states: torch.Tensor):
        x = hidden_states
        orig_dtype = x.dtype
        if x.dtype != self.gate.weight.dtype:
            x = x.to(self.gate.weight.dtype)
        router_logits = self.gate(x)
        routing_weights = torch.softmax(router_logits, dim=1, dtype=torch.float32)
        routing_weights, selected = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(orig_dtype)

        loc = selected - self.expert_base                        # [T, k]
        is_local = (loc >= 0) & (loc < self.local_experts)
        loc_c = loc.clamp(0, self.local_experts - 1).to(torch.int32)

        if self.offload:
            out = self._forward_offload(x, orig_dtype, loc, loc_c, is_local, routing_weights)
        else:
            out = self._forward_resident(x, orig_dtype, loc, loc_c, is_local, routing_weights)

        if self._has_shared and self.tp_rank == 0:
            shared = self.shared_expert.down_proj(
                F.silu(self.shared_expert.gate_proj(x)) * self.shared_expert.up_proj(x))
            shared = shared * torch.sigmoid(self.shared_expert_gate(x))
            out = out + shared.to(orig_dtype)
        if self.tp_size > 1:
            dist.all_reduce(out)
        return out.to(orig_dtype), router_logits


# ---------------------------------------------------------------------------
# Decoder layer / model
# ---------------------------------------------------------------------------

class Qwen4ExpAttention(Qwen3_5Attention):
    """Gated full attention + QSA indexer weights (indexer approximated by a
    sliding window of indexer_budget tokens - EXACT at ctx<=2048)."""

    def __init__(self, config, layer_idx: int, max_position=None):
        super().__init__(config, max_position=max_position)
        self.indexer = nn.Module()
        self.indexer.index_qk_proj = FP16Linear(
            config.hidden_size,
            (config.indexer_n_heads + config.indexer_kv_heads) * config.indexer_head_dim,
            bias=False)
        self.indexer.q_layernorm = GroupedGemmaRMSNorm(config.indexer_head_dim, None, eps=config.rms_norm_eps)
        self.indexer.k_layernorm = GroupedGemmaRMSNorm(config.indexer_head_dim, None, eps=config.rms_norm_eps)


class Qwen4ExpDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int, max_position=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.layer_type in ("full_attention", "indexed_attention", "qwen_sparse_attention"):
            self.self_attn = Qwen4ExpAttention(config, layer_idx, max_position=max_position)
        else:
            raise ValueError(f"Unknown layer type: {self.layer_type}")
        self.mlp = Qwen4ExpSparseMoeBlock(config, layer_idx)
        ple_layer_index = (config.ple_layer_ids.index(layer_idx + 1)
                           if (layer_idx + 1) in config.ple_layer_ids else None)
        self.ple = (Qwen4ExpPLELayer(config, layer_idx, ple_layer_index)
                    if ple_layer_index is not None else None)
        self.attn_hyper_connection = Qwen4ExpGatedResidual(config)
        self.mlp_hyper_connection = Qwen4ExpGatedResidual(config)

    def forward(self, positions, hidden_states, bounds, is_prefill):
        if self.ple is not None:
            hidden_states = hidden_states + self.ple(hidden_states, bounds, is_prefill)
        block_input, hyper, injection = self.attn_hyper_connection.mix(hidden_states)
        if self.layer_type == "linear_attention":
            out = self.linear_attn(block_input)
        else:
            out = self.self_attn(positions, block_input)
        hidden_states = Qwen4ExpGatedResidual.combine(hyper, out, injection)
        block_input, hyper, injection = self.mlp_hyper_connection.mix(hidden_states)
        out, _ = self.mlp(block_input)
        hidden_states = Qwen4ExpGatedResidual.combine(hyper, out, injection)
        return hidden_states


class Qwen4ExpTextModel(nn.Module):
    def __init__(self, config, max_position=None):
        super().__init__()
        from adallm.ops.embed_head import VocabParallelEmbedding
        self.hc_count = config.hc_count
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen4ExpDecoderLayer(config, i, max_position=max_position)
             for i in range(config.num_hidden_layers)])
        self.hyper_connection_mixer = Qwen4ExpGatedResidual(config, use_combine=False)

    def forward(self, input_ids, positions):
        context = get_context()
        is_prefill = context.is_prefill
        bounds = None
        if is_prefill:
            cu_host = context.cu_q_list
            slots_host = context.slots_list
            if cu_host is not None and slots_host is not None and len(cu_host) == len(slots_host) + 1:
                bounds = [(cu_host[i], cu_host[i + 1], slots_host[i]) for i in range(len(slots_host))]
            else:
                cu = context.cu_seqlens_q
                slots = context.delta_slots
                bounds = [(int(cu[i]), int(cu[i + 1]), int(slots[i])) for i in range(slots.shape[0])]

        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states.repeat(1, self.hc_count)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states, bounds, is_prefill)
        mixed, _, _ = self.hyper_connection_mixer.mix(hidden_states)
        return mixed


class Qwen4ExpModel(nn.Module):
    """Wrapper matching checkpoint prefix `model.language_model.*`."""

    def __init__(self, config, max_position=None):
        super().__init__()
        self.language_model = Qwen4ExpTextModel(config, max_position=max_position)

    def forward(self, input_ids, positions):
        return self.language_model(input_ids, positions)


class Qwen4ExpForCausalLM(nn.Module):
    """Loader mapping (handled by an early branch in infra/loader.py):
    - expert tensors ...mlp.experts.<gid>.<proj>.<weight|weight_scale|weight_scale_2>
      -> ...mlp._<gate|up|down>_<suffix> EP params; remote experts skipped
         before materialization (half the 297k tensors never get read).
    - PLE shard tensors ...ngram_embedding.shard_i.weight -> skipped
      (the 51GB table is read directly from the file via np.memmap).
    - PLE scale ...ngram_embedding.weight_scale -> python float on the
      NgramEmbedding (used for GPU dequant of gathered rows).
    """

    _EXPERT_RE = re.compile(
        r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
        r"(gate_proj|up_proj|down_proj)\.(weight|weight_scale|weight_scale_2)$")
    _EXPERT_AUX_RE = re.compile(
        r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
        r"(gate_proj|up_proj|down_proj)\.input_scale$")
    _PLE_SHARD_RE = re.compile(
        r"^model\.language_model\.layers\.(\d+)\.ple\.ple_embedding\.ngram_embedding\.shard_\d+\.weight$")
    _PLE_SCALE_RE = re.compile(
        r"^model\.language_model\.layers\.(\d+)\.ple\.ple_embedding\.ngram_embedding\.weight_scale$")
    _ATTR = {"gate_proj": "_gate", "up_proj": "_up", "down_proj": "_down"}

    def __init__(self, config, max_position=None):
        super().__init__()
        self.config = config
        self.model = Qwen4ExpModel(config, max_position=max_position)
        from adallm.ops.embed_head import ParallelLMHead
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight.data = self.model.language_model.embed_tokens.weight.data
        self._ple_scratch: torch.Tensor | None = None

    # -- loader hooks (called from adallm/infra/loader.py) -------------------
    def q4e_map(self, name: str):
        """None = not ours; ('expert', layer, gid, new_name);
        ('ple_skip',); ('ple_scale', layer)."""
        m = self._EXPERT_RE.match(name)
        if m:
            layer_idx, gid, proj, suffix = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
            attr = {"weight_scale": "scale", "weight_scale_2": "scale_2", "weight": "weight"}[suffix]
            new = f"model.language_model.layers.{layer_idx}.mlp.{self._ATTR[proj]}_{attr}"
            return ("expert", layer_idx, gid, new)
        if self._EXPERT_AUX_RE.match(name) or self._PLE_SHARD_RE.match(name):
            return ("ple_skip",)
        m = self._PLE_SCALE_RE.match(name)
        if m:
            return ("ple_scale", int(m.group(1)))
        return None

    def _moe_block(self, layer_idx: int) -> "Qwen4ExpSparseMoeBlock":
        return self.model.language_model.layers[layer_idx].mlp

    def expert_is_local(self, layer_idx: int, gid: int) -> bool:
        return self._moe_block(layer_idx).expert_is_local(gid)

    def load_expert_tensor(self, new_name: str, gid: int, tensor: torch.Tensor):
        param = self.get_parameter(new_name)
        param.weight_loader(param, tensor, gid)

    def load_ple_scale(self, layer_idx: int, tensor: torch.Tensor):
        emb = self.model.language_model.layers[layer_idx].ple.ple_embedding
        emb.weight_scale = float(tensor.float().reshape(()).item())

    # -- runtime PLE wiring (ModelRunner) -------------------------------------
    def ple_layers(self):
        return [l.ple for l in self.model.language_model.layers if l.ple is not None]

    def attach_ple_file(self, path: str):
        for ple in self.ple_layers():
            ple.ple_embedding.attach_file(path)

    def allocate_ple(self, num_slots: int, max_tokens: int, device, dtype):
        for ple in self.ple_layers():
            ple.allocate_state_pool(num_slots, max_tokens, device, dtype)
        heads_dim = self.ple_layers()[0].ple_embedding.embedding_dim
        self._ple_scratch = torch.empty(max_tokens * heads_dim, dtype=torch.uint8,
                                        device="cpu").pin_memory()

    def ple_prepare(self, segments):
        """Host-side PLE embedding prep (all ranks run the same segments)."""
        for ple in self.ple_layers():
            ple.ple_embedding.prepare_step(segments, ple.stage, self._ple_scratch)

    def ple_reset_slot(self, slot: int):
        for ple in self.ple_layers():
            ple.reset_slot(slot)

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)
