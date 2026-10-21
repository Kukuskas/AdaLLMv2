import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from adallm.infra.context import get_context
from adallm.infra.quant import get_quant_config


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


@triton.jit
def fp8_kv_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    block_table_ptr,
    context_lens_ptr,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_od,
    stride_bt,
    B,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    Q_PER_KV: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // NUM_HEADS
    h = pid % NUM_HEADS
    if b >= B:
        return
    kv_h = h // Q_PER_KV
    ctx_len = tl.load(context_lens_ptr + b)
    if ctx_len == 0:
        return

    d = tl.arange(0, HEAD_DIM)
    q = tl.load(q_ptr + b * stride_qb + h * stride_qh + d * stride_qd, mask=d < HEAD_DIM, other=0).to(tl.float16)

    m = tl.full((), -float("inf"), tl.float32)
    l = tl.zeros((), tl.float32)
    acc = tl.zeros((HEAD_DIM,), tl.float32)

    num_blocks = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    for blk in range(0, MAX_BLOCKS):
        blk_valid = blk < num_blocks
        block_id = tl.load(block_table_ptr + b * stride_bt + blk, mask=blk_valid, other=0)
        token_base = blk * BLOCK_SIZE
        for off in tl.static_range(0, BLOCK_SIZE, BLOCK_T):
            t = tl.arange(0, BLOCK_T)
            token_idx = token_base + off + t
            mask_t = (token_idx < ctx_len) & blk_valid & (block_id >= 0)
            k = tl.load(
                k_ptr + block_id * stride_kb + (off + t) * stride_kt + kv_h * stride_kh + d[None, :] * stride_kd,
                mask=mask_t[:, None] & (d[None, :] < HEAD_DIM),
                other=0.0,
            ).to(tl.float16)
            v = tl.load(
                v_ptr + block_id * stride_vb + (off + t) * stride_vt + kv_h * stride_vh + d[None, :] * stride_vd,
                mask=mask_t[:, None] & (d[None, :] < HEAD_DIM),
                other=0.0,
            ).to(tl.float16)
            scores = tl.sum(k * q[None, :], axis=1).to(tl.float32)
            scores = tl.where(mask_t, scores, -float("inf"))
            block_m = tl.max(scores, axis=0)
            m_new = tl.maximum(m, block_m)
            exp_scores = tl.exp(scores - m_new)
            l = l * tl.exp(m - m_new) + tl.sum(exp_scores, axis=0)
            acc = acc * tl.exp(m - m_new) + tl.sum(exp_scores[:, None] * v.to(tl.float32), axis=0)
            m = m_new

    out = acc / l
    tl.store(out_ptr + b * stride_ob + h * stride_oh + d * stride_od, out.to(tl.float16), mask=d < HEAD_DIM)


def _make_fp8_decode_autotune_configs():
    # Keep a small, safe config set to avoid shared-memory OOR during autotune.
    # NOTE: with HEAD_DIM=256 (Qwen3.8 family), BLOCK_T=64 @ num_stages=2
    # requires ~133KB shared memory > 99KB RTX 4090 limit. BLOCK_T=32 and/or
    # num_stages=1 stay within budget.
    return [
        triton.Config({"BLOCK_T": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_T": 32}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_T": 64}, num_warps=4, num_stages=1),
    ]


FP8_DECODE_AUTOTUNE_CONFIGS = _make_fp8_decode_autotune_configs()


@triton.autotune(
    configs=FP8_DECODE_AUTOTUNE_CONFIGS,
    key=["BLOCK_SIZE", "HEAD_DIM", "Q_PER_KV", "BLOCK_Q"],
)
@triton.jit
def fp8_kv_decode_group_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    block_table_ptr,
    context_lens_ptr,
    scale,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_od,
    stride_bt,
    B,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    Q_PER_KV: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_T: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // NUM_KV_HEADS
    kv_h = pid % NUM_KV_HEADS
    if b >= B:
        return
    ctx_len = tl.load(context_lens_ptr + b)
    if ctx_len == 0:
        return
    if WINDOW_SIZE > 0:
        start = tl.maximum(ctx_len - WINDOW_SIZE, 0)
        start_block = start // BLOCK_SIZE
    else:
        start = 0
        start_block = 0

    q_idx = tl.arange(0, BLOCK_Q)
    h = kv_h * Q_PER_KV + q_idx
    mask_h = (q_idx < Q_PER_KV) & (h < NUM_HEADS)
    d = tl.arange(0, HEAD_DIM)
    q = tl.load(
        q_ptr + b * stride_qb + h[:, None] * stride_qh + d[None, :] * stride_qd,
        mask=mask_h[:, None] & (d[None, :] < HEAD_DIM),
        other=0,
    ).to(tl.float16)

    m = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
    l = tl.zeros((BLOCK_Q,), tl.float32)
    acc = tl.zeros((BLOCK_Q, HEAD_DIM), tl.float32)

    num_blocks = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    for blk in range(0, MAX_BLOCKS):
        blk_valid = blk < num_blocks
        if WINDOW_SIZE > 0:
            blk_valid = blk_valid & (blk >= start_block)
        block_id = tl.load(block_table_ptr + b * stride_bt + blk, mask=blk_valid, other=0)
        token_base = blk * BLOCK_SIZE
        for off in tl.static_range(0, BLOCK_SIZE, BLOCK_T):
            t = tl.arange(0, BLOCK_T)
            token_idx = token_base + off + t
            mask_t = (token_idx < ctx_len) & blk_valid & (block_id >= 0)
            if WINDOW_SIZE > 0:
                mask_t = mask_t & (token_idx >= start)
            k_t = tl.load(
                k_ptr + block_id * stride_kb + kv_h * stride_kh + (off + t)[None, :] * stride_kt + d[:, None] * stride_kd,
                mask=mask_t[None, :] & (d[:, None] < HEAD_DIM),
                other=0.0,
            ).to(tl.float16)
            v_t = tl.load(
                v_ptr + block_id * stride_vb + kv_h * stride_vh + (off + t)[:, None] * stride_vt + d[None, :] * stride_vd,
                mask=mask_t[:, None] & (d[None, :] < HEAD_DIM),
                other=0.0,
            ).to(tl.float16)
            scores = tl.dot(q, k_t) * scale
            scores = tl.where(mask_t[None, :], scores, -float("inf"))
            scores = tl.where(mask_h[:, None], scores, -float("inf"))
            block_m = tl.max(scores, axis=1)
            m_new = tl.maximum(m, block_m)
            exp_scores = tl.exp(scores - m_new[:, None])
            exp_scores_fp16 = exp_scores.to(tl.float16)
            l = l * tl.exp(m - m_new) + tl.sum(exp_scores, axis=1)
            acc = acc * tl.exp(m - m_new)[:, None] + tl.dot(exp_scores_fp16, v_t)
            m = m_new

    out = acc / l[:, None]
    tl.store(
        out_ptr + b * stride_ob + h[:, None] * stride_oh + d[None, :] * stride_od,
        out.to(tl.float16),
        mask=mask_h[:, None] & (d[None, :] < HEAD_DIM),
    )


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert slot_mapping.numel() == N
    if k_cache.stride(1) == D and v_cache.stride(1) == D:
        store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)
        return
    # Fallback for unexpected layouts.
    slot = slot_mapping
    valid = slot != -1
    if torch.any(valid):
        slots = slot[valid].to(torch.int64)
        k_cache.view(-1, D)[slots] = key[valid].to(k_cache.dtype)
        v_cache.view(-1, D)[slots] = value[valid].to(v_cache.dtype)


def _gather_kv_from_blocks(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Layout: [block, token, kv_head, head_dim] for all dtypes.
    block_size = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2]
    head_dim = k_cache.shape[3]
    total_tokens = int(context_lens.sum().item())
    device = k_cache.device
    k_out = torch.empty((total_tokens, num_kv_heads, head_dim), device=device, dtype=torch.float16)
    v_out = torch.empty_like(k_out)
    cu_seqlens_k = [0]
    offset = 0
    for i in range(block_tables.shape[0]):
        ctx_len = int(context_lens[i].item())
        if ctx_len == 0:
            cu_seqlens_k.append(cu_seqlens_k[-1])
            continue
        num_blocks = (ctx_len + block_size - 1) // block_size
        blocks = block_tables[i, :num_blocks].to(torch.int64)
        kv_k = k_cache.index_select(0, blocks).to(torch.float16)
        kv_v = v_cache.index_select(0, blocks).to(torch.float16)
        kv_k = kv_k.reshape(num_blocks * block_size, num_kv_heads, head_dim)[:ctx_len]
        kv_v = kv_v.reshape(num_blocks * block_size, num_kv_heads, head_dim)[:ctx_len]
        k_out[offset:offset + ctx_len] = kv_k
        v_out[offset:offset + ctx_len] = kv_v
        offset += ctx_len
        cu_seqlens_k.append(offset)
    cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, device=device)
    return k_out, v_out, cu_seqlens_k


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        sliding_window: int | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        orig_dtype = q.dtype
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        k_cache_compute, v_cache_compute = k_cache, v_cache
        is_fp8_cache = k_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) or v_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        if is_fp8_cache:
            compute_dtype = torch.float16
        else:
            compute_dtype = k_cache.dtype if k_cache.dtype != q.dtype else q.dtype
        if k.dtype != compute_dtype:
            k = k.to(compute_dtype)
        if v.dtype != compute_dtype:
            v = v.to(compute_dtype)
        if q.dtype != compute_dtype:
            q = q.to(compute_dtype)
        if not is_fp8_cache:
            if k_cache.dtype != compute_dtype:
                k_cache_compute = k_cache.to(compute_dtype)
            if v_cache.dtype != compute_dtype:
                v_cache_compute = v_cache.to(compute_dtype)
        window_size = (-1, -1)
        window_tokens = 0
        if self.sliding_window is not None and self.sliding_window > 0:
            window_size = (self.sliding_window, 0)
            window_tokens = self.sliding_window
        if context.is_prefill:
            cu_seqlens_k = context.cu_seqlens_k
            if context.block_tables is not None:    # prefix cache
                if is_fp8_cache:
                    k_cache_compute, v_cache_compute, cu_seqlens_k = _gather_kv_from_blocks(
                        k_cache, v_cache, context.block_tables, context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1]
                    )
                    k, v = k_cache_compute, v_cache_compute
                else:
                    k, v = k_cache_compute, v_cache_compute
                    cu_seqlens_k = context.cu_seqlens_k
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                window_size=window_size,
                block_table=None if is_fp8_cache else context.block_tables,
            )
        else:    # decode
            if is_fp8_cache:
                qcfg = get_quant_config()
                if not qcfg.fp8_decode_kernel_enabled:
                    raise RuntimeError("FP8 KV cache requires fp8_decode_kernel_enabled=True.")
                try:
                    bs = q.shape[0]
                    q_contig = q.contiguous()
                    out = torch.empty_like(q_contig, dtype=torch.float16)
                    block_size = k_cache.shape[1]
                    head_dim = q_contig.shape[-1]
                    num_heads = q_contig.shape[1]
                    num_kv_heads = k_cache.shape[2]
                    q_per_kv = num_heads // num_kv_heads
                    block_q = 1 << (q_per_kv - 1).bit_length()
                    grid = (bs * num_kv_heads,)
                    fp8_kv_decode_group_kernel[grid](
                        q_contig,
                        k_cache,
                        v_cache,
                        out,
                        context.block_tables,
                        context.context_lens,
                        self.scale,
                        q_contig.stride(0),
                        q_contig.stride(1),
                        q_contig.stride(2),
                        k_cache.stride(0),
                        k_cache.stride(1),
                        k_cache.stride(2),
                        k_cache.stride(3),
                        v_cache.stride(0),
                        v_cache.stride(1),
                        v_cache.stride(2),
                        v_cache.stride(3),
                        out.stride(0),
                        out.stride(1),
                        out.stride(2),
                        context.block_tables.stride(0),
                        bs,
                        MAX_BLOCKS=context.block_tables.shape[1],
                        BLOCK_SIZE=block_size,
                        HEAD_DIM=head_dim,
                        NUM_HEADS=num_heads,
                        NUM_KV_HEADS=num_kv_heads,
                        Q_PER_KV=q_per_kv,
                        BLOCK_Q=block_q,
                        WINDOW_SIZE=window_tokens,
                    )
                    o = out
                except Exception as exc:
                    if torch.cuda.is_current_stream_capturing():
                        raise
                    raise RuntimeError(
                        "FP8 KV decode kernel failed; no fallback is allowed in this runtime."
                    ) from exc
            else:
                o = flash_attn_with_kvcache(
                    q.unsqueeze(1),
                    k_cache_compute,
                    v_cache_compute,
                    cache_seqlens=context.context_lens,
                    block_table=context.block_tables,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=window_size,
                )
        if o.dtype != orig_dtype:
            o = o.to(orig_dtype)
        return o
