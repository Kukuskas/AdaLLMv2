import os

import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from adallm.infra.context import get_context
from adallm.infra.quant import get_quant_config

# Split-K flash-decode for fp8 KV (round2-c): opt-in via ADALLM_ATTN_SPLITK=1;
# the original fp8_kv_decode_group_kernel remains the default fallback.
_SPLITK_ENABLED = os.environ.get("ADALLM_ATTN_SPLITK", "0") == "1"
# Round8 L1a: paged verify attention — small fp8 prefill spans (MTP verify/
# replay, <=8 rows) attend directly over block_tables via the splitk decode
# kernel instead of gathering the whole context KV to fp16 (~2GB @28K).
_PAGED_VERIFY = os.environ.get("ADALLM_PAGED_VERIFY", "0") == "1"


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


# ---------------------------------------------------------------------------
# Split-K flash-decode kernel (round2-c). Parallelizes the KV scan across
# MAX_SPLIT programs per (batch, kv_head); each program computes an online-
# softmax partial (m, l, acc) over its KV block chunk; a tiny reduce kernel
# merges partials. Split count adapts INSIDE the kernel from the on-device
# context_lens (CUDA-graph safe). Enabled via ADALLM_ATTN_SPLITK=1.
# ---------------------------------------------------------------------------
@triton.jit
def fp8_kv_decode_splitk_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    pm_ptr,
    pl_ptr,
    pacc_ptr,
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
    stride_pm_b,
    stride_pm_h,
    stride_pacc_b,
    stride_pacc_h,
    stride_pacc_s,
    stride_pacc_q,
    stride_pacc_d,
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
    MAX_SPLIT: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    TOKENS_PER_SPLIT_MIN: tl.constexpr,
):
    pid_b = tl.program_id(0)
    kv_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    q_idx0 = tl.arange(0, BLOCK_Q)
    h0 = kv_h * Q_PER_KV + q_idx0
    mask_h0 = (q_idx0 < Q_PER_KV) & (h0 < NUM_HEADS)
    d0 = tl.arange(0, HEAD_DIM)
    blk_lo = 0
    blk_hi = -1
    tok_lo = 0
    tok_hi = -1
    empty = 1
    if pid_b < B:
        ctx_len = tl.load(context_lens_ptr + pid_b)
        if ctx_len > 0:
            if WINDOW_SIZE > 0:
                start = tl.maximum(ctx_len - WINDOW_SIZE, 0)
                start_block = start // BLOCK_SIZE
            else:
                start = 0
                start_block = 0
            end_block = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
            num_tokens = ctx_len - start
            num_blocks = end_block - start_block
            nsplit = (num_tokens + TOKENS_PER_SPLIT_MIN - 1) // TOKENS_PER_SPLIT_MIN
            nsplit = tl.minimum(nsplit, MAX_SPLIT)
            chunk_blocks = (num_blocks + nsplit - 1) // nsplit
            blk_lo = start_block + pid_s * chunk_blocks
            blk_hi = tl.minimum(blk_lo + chunk_blocks, end_block)
            tok_lo = tl.maximum(start, blk_lo * BLOCK_SIZE)
            tok_hi = tl.minimum(ctx_len, blk_hi * BLOCK_SIZE)
            if (pid_s < nsplit) & (blk_lo < blk_hi) & (tok_lo < tok_hi):
                empty = 0

    if empty == 1:
        # Identity partial so the reducer never sees uninitialized memory.
        m_id = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
        l_id = tl.zeros((BLOCK_Q,), tl.float32)
        a_id = tl.zeros((BLOCK_Q, HEAD_DIM), tl.float32)
        tl.store(pm_ptr + pid_b * stride_pm_b + kv_h * stride_pm_h + pid_s * BLOCK_Q + q_idx0, m_id, mask=mask_h0)
        tl.store(pl_ptr + pid_b * stride_pm_b + kv_h * stride_pm_h + pid_s * BLOCK_Q + q_idx0, l_id, mask=mask_h0)
        tl.store(
            pacc_ptr + pid_b * stride_pacc_b + kv_h * stride_pacc_h + pid_s * stride_pacc_s
            + q_idx0[:, None] * stride_pacc_q + d0[None, :] * stride_pacc_d,
            a_id,
            mask=mask_h0[:, None] & (d0[None, :] < HEAD_DIM),
        )
        return

    q_idx = tl.arange(0, BLOCK_Q)
    h = kv_h * Q_PER_KV + q_idx
    mask_h = (q_idx < Q_PER_KV) & (h < NUM_HEADS)
    d = tl.arange(0, HEAD_DIM)
    q = tl.load(
        q_ptr + pid_b * stride_qb + h[:, None] * stride_qh + d[None, :] * stride_qd,
        mask=mask_h[:, None] & (d[None, :] < HEAD_DIM),
        other=0,
    ).to(tl.float16)

    m = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
    l = tl.zeros((BLOCK_Q,), tl.float32)
    acc = tl.zeros((BLOCK_Q, HEAD_DIM), tl.float32)

    for blk in range(blk_lo, blk_hi):
        blk_valid = blk < MAX_BLOCKS
        block_id = tl.load(block_table_ptr + pid_b * stride_bt + blk, mask=blk_valid, other=0)
        token_base = blk * BLOCK_SIZE
        for off in tl.static_range(0, BLOCK_SIZE, BLOCK_T):
            t = tl.arange(0, BLOCK_T)
            token_idx = token_base + off + t
            mask_t = (token_idx >= tok_lo) & (token_idx < tok_hi) & blk_valid & (block_id >= 0)
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
            m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            exp_scores = tl.exp(scores - m_safe[:, None])
            exp_scores = tl.where(m_new[:, None] == -float("inf"), 0.0, exp_scores)
            exp_scores_fp16 = exp_scores.to(tl.float16)
            alpha = tl.exp(m - m_safe)
            l = l * alpha + tl.sum(exp_scores, axis=1)
            acc = acc * alpha[:, None] + tl.dot(exp_scores_fp16, v_t)
            m = m_new

    tl.store(pm_ptr + pid_b * stride_pm_b + kv_h * stride_pm_h + pid_s * BLOCK_Q + q_idx, m, mask=mask_h)
    tl.store(pl_ptr + pid_b * stride_pm_b + kv_h * stride_pm_h + pid_s * BLOCK_Q + q_idx, l, mask=mask_h)
    tl.store(
        pacc_ptr + pid_b * stride_pacc_b + kv_h * stride_pacc_h + pid_s * stride_pacc_s
        + q_idx[:, None] * stride_pacc_q + d[None, :] * stride_pacc_d,
        acc,
        mask=mask_h[:, None] & (d[None, :] < HEAD_DIM),
    )


@triton.jit
def splitk_reduce_kernel(
    pm_ptr,
    pl_ptr,
    pacc_ptr,
    out_ptr,
    stride_pm_b,
    stride_pm_h,
    stride_pacc_b,
    stride_pacc_h,
    stride_pacc_s,
    stride_pacc_q,
    stride_pacc_d,
    stride_ob,
    stride_oh,
    stride_od,
    B,
    HEAD_DIM: tl.constexpr,
    Q_PER_KV: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    MAX_SPLIT: tl.constexpr,
):
    pid_b = tl.program_id(0)
    h = tl.program_id(1)
    if pid_b >= B:
        return
    kv_h = h // Q_PER_KV
    local_q = h % Q_PER_KV
    d = tl.arange(0, HEAD_DIM)

    pm_base = pid_b * stride_pm_b + kv_h * stride_pm_h + local_q
    pacc_base = pid_b * stride_pacc_b + kv_h * stride_pacc_h + local_q * stride_pacc_q

    m_max = -float("inf")
    for s in tl.static_range(0, MAX_SPLIT):
        m_s = tl.load(pm_ptr + pm_base + s * BLOCK_Q)
        m_max = tl.maximum(m_max, m_s)

    l_tot = 0.0
    acc = tl.zeros((HEAD_DIM,), tl.float32)
    for s in tl.static_range(0, MAX_SPLIT):
        m_s = tl.load(pm_ptr + pm_base + s * BLOCK_Q)
        l_s = tl.load(pl_ptr + pm_base + s * BLOCK_Q)
        a_s = tl.load(pacc_ptr + pacc_base + s * stride_pacc_s + d * stride_pacc_d)
        w = tl.exp(m_s - m_max)
        l_tot += l_s * w
        acc += a_s * w

    out = acc / l_tot
    tl.store(out_ptr + pid_b * stride_ob + h * stride_oh + d * stride_od, out.to(tl.float16), mask=d < HEAD_DIM)


def fp8_kv_decode_splitk(
    q,
    k_cache,
    v_cache,
    block_tables,
    context_lens,
    scale,
    window_tokens=0,
    max_split=64,
    block_t=32,
    tokens_per_split_min=1024,
):
    """Split-K flash-decode over fp8 KV cache. Same contract as
    fp8_kv_decode_group_kernel's call site: returns [bs, num_heads, head_dim]
    fp16 with GQA broadcast folded in."""
    bs, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    block_size = k_cache.shape[1]
    q_per_kv = num_heads // num_kv_heads
    block_q = 1 << (q_per_kv - 1).bit_length()
    dev = q.device
    pm = torch.empty(bs, num_kv_heads, max_split, block_q, device=dev, dtype=torch.float32)
    pl = torch.empty(bs, num_kv_heads, max_split, block_q, device=dev, dtype=torch.float32)
    pacc = torch.empty(bs, num_kv_heads, max_split, block_q, head_dim, device=dev, dtype=torch.float32)
    out = torch.empty(bs, num_heads, head_dim, device=dev, dtype=torch.float16)
    fp8_kv_decode_splitk_kernel[(bs, num_kv_heads, max_split)](
        q,
        k_cache,
        v_cache,
        pm,
        pl,
        pacc,
        block_tables,
        context_lens,
        scale,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        pm.stride(0), pm.stride(1),
        pacc.stride(0), pacc.stride(1), pacc.stride(2), pacc.stride(3), pacc.stride(4),
        block_tables.stride(0),
        bs,
        MAX_BLOCKS=block_tables.shape[1],
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        NUM_HEADS=num_heads,
        NUM_KV_HEADS=num_kv_heads,
        Q_PER_KV=q_per_kv,
        BLOCK_Q=block_q,
        BLOCK_T=block_t,
        MAX_SPLIT=max_split,
        WINDOW_SIZE=window_tokens,
        TOKENS_PER_SPLIT_MIN=tokens_per_split_min,
        num_warps=4,
        num_stages=1,
    )
    splitk_reduce_kernel[(bs, num_heads)](
        pm, pl, pacc, out,
        pm.stride(0), pm.stride(1),
        pacc.stride(0), pacc.stride(1), pacc.stride(2), pacc.stride(3), pacc.stride(4),
        out.stride(0), out.stride(1), out.stride(2),
        bs,
        HEAD_DIM=head_dim,
        Q_PER_KV=q_per_kv,
        NUM_HEADS=num_heads,
        BLOCK_Q=block_q,
        MAX_SPLIT=max_split,
        num_warps=4,
        num_stages=1,
    )
    return out



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
            # Round8-B (lever 1): shadow fp16 mirror of the fp8 KV cache.
            # KV is append-only; store the UNquantized k/v into a parallel fp16
            # cache so prefill-style forwards (MTP verify) can use paged flash
            # attention over block_tables instead of the whole-context gather
            # (_gather_kv_from_blocks: ~2GB copy + .item() loops at 28K ctx).
            # Env-gated: ADALLM_KV_SHADOW=1.
            sk = getattr(self, "k_shadow", None)
            if sk is not None and sk.numel():
                store_kvcache(k, v, sk, self.v_shadow, context.slot_mapping)
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
                sk = getattr(self, "k_shadow", None)
                if (_PAGED_VERIFY and _SPLITK_ENABLED and is_fp8_cache
                        and q.shape[0] <= 8
                        and context.cu_seqlens_q.numel() == 2):
                    # Round8 L1a: paged verify — small single-seq fp8 spans
                    # (MTP verify/replay continuation) attend over block_tables
                    # via the splitk decode kernel with per-row lens
                    # start+1+j (row j's own KV row was just stored by
                    # store_kvcache above). No gather, no host sync, no extra
                    # memory (unlike the fp16 shadow mirror).
                    n_q = q.shape[0]
                    base = (context.cu_seqlens_k[1] - n_q).reshape(1)
                    lens = base + torch.arange(1, n_q + 1, device=q.device,
                                               dtype=torch.int32)
                    bt = context.block_tables[:1].expand(n_q, -1)
                    o = fp8_kv_decode_splitk(
                        q.contiguous(), k_cache, v_cache, bt, lens, self.scale,
                        window_tokens=window_tokens)
                    if o.dtype != orig_dtype:
                        o = o.to(orig_dtype)
                    return o
                if is_fp8_cache and sk is not None and sk.numel():
                    # Round8-B: paged flash over the fp16 shadow mirror — no
                    # gather, no .item(), graph-friendly. The newly computed
                    # k/v for this forward's own tokens are already stored
                    # into the shadow by store_kvcache above, so q/k/v for
                    # flash = shadow cache entirely (append-only semantics).
                    k_cache_compute, v_cache_compute = sk, self.v_shadow
                    cu_seqlens_k = context.cu_seqlens_k
                    k, v = k_cache_compute, v_cache_compute
                    seqused_k = None
                    if cu_seqlens_k is not None and cu_seqlens_k.numel() >= 2:
                        # vllm flash with block_table needs seqused_k INSTEAD of cu_seqlens_k
                        seqused_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
                    o = flash_attn_varlen_func(
                        q, k, v,
                        max_seqlen_q=context.max_seqlen_q,
                        cu_seqlens_q=context.cu_seqlens_q,
                        max_seqlen_k=context.max_seqlen_k,
                        seqused_k=seqused_k,
                        softmax_scale=self.scale,
                        causal=True,
                        window_size=(list(window_size) if isinstance(window_size, tuple) else window_size),
                        block_table=context.block_tables,
                    )
                    if o.dtype != orig_dtype:
                        o = o.to(orig_dtype)
                    return o
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
                    block_size = k_cache.shape[1]
                    head_dim = q_contig.shape[-1]
                    num_heads = q_contig.shape[1]
                    num_kv_heads = k_cache.shape[2]
                    q_per_kv = num_heads // num_kv_heads
                    if _SPLITK_ENABLED:
                        o = fp8_kv_decode_splitk(
                            q_contig,
                            k_cache,
                            v_cache,
                            context.block_tables,
                            context.context_lens,
                            self.scale,
                            window_tokens=window_tokens,
                        )
                        if o.dtype != orig_dtype:
                            o = o.to(orig_dtype)
                        return o
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
