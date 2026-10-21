import os
import triton
import triton.language as tl
import torch


def _make_fp4_lut() -> torch.Tensor:
    lut = []
    for code in range(16):
        sign = -1.0 if (code & 0x8) else 1.0
        exp = (code >> 1) & 0x3
        mant = code & 0x1
        if exp == 0:
            val = 0.0 if mant == 0 else 0.5
        else:
            val = (1.0 + mant * 0.5) * (2.0 ** (exp - 1))
        lut.append(sign * val)
    return torch.tensor(lut, dtype=torch.float16)


_FP4_LUT = _make_fp4_lut()

_FP8_ENV = os.getenv("NVFP4_FP8", "0").strip().lower()
_FP8_ENABLED = _FP8_ENV not in ("0", "false", "off")
_FP8_FORMAT = os.getenv("NVFP4_FP8_FORMAT", "e4nv").strip().lower()
_FP8_DTYPE = {
    "e4nv": getattr(tl, "float8e4nv", None),
    "e4m3": getattr(tl, "float8e4nv", None),
    "e5": getattr(tl, "float8e5", None),
    "e5m2": getattr(tl, "float8e5", None),
    "e4b15": getattr(tl, "float8e4b15", None),
    "e4b8": getattr(tl, "float8e4b8", None),
    "e5b16": getattr(tl, "float8e5b16", None),
}.get(_FP8_FORMAT, getattr(tl, "float8e4nv", None))
_FP8_A_SCALE_ENV = os.getenv("NVFP4_FP8_A_SCALE", "1").strip().lower()
_FP8_A_SCALE = _FP8_A_SCALE_ENV not in ("0", "false", "off")
_FP8_B_SCALE_ENV = os.getenv("NVFP4_FP8_B_SCALE", "0").strip().lower()
_FP8_B_SCALE = _FP8_B_SCALE_ENV not in ("0", "false", "off")
_FP8_BCOL_ENV = os.getenv("NVFP4_FP8_BCOL", "0").strip().lower()
_FP8_BCOL = _FP8_BCOL_ENV not in ("0", "false", "off")


def _fp8_supported() -> bool:
    if _FP8_DTYPE is None:
        return False
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) >= (8, 9)


_FP4_LUT_BY_DEVICE: dict = {}


def _fp4_lut_on_device(device: torch.device) -> torch.Tensor:
    key = (device.type, device.index)
    lut = _FP4_LUT_BY_DEVICE.get(key)
    if lut is None:
        lut = _FP4_LUT.to(device=device)
        _FP4_LUT_BY_DEVICE[key] = lut
    return lut


def warm_cudagraph_buffers(device: torch.device) -> None:
    """Materialize all lazily-created device constants (LUT) BEFORE CUDA graph
    capture. Otherwise the first .to(device) copy inside the capture raises
    'Cannot copy between CPU and CUDA tensors during CUDA graph capture'."""
    _fp4_lut_on_device(device)


@triton.jit
def nvfp4_gemm_kernel(
    A_ptr,
    B_ptr,
    S_ptr,
    S2_ptr,
    LUT_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_sn,
    stride_sk,
    stride_cm,
    stride_cn,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    scale2 = tl.load(S2_ptr).to(tl.float16)

    k = 0
    while k < K:
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0,
        )
        a = a.to(tl.float16)

        packed_idx = offs_k >> 1
        b_packed = tl.load(
            B_ptr + packed_idx[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0,
        )
        low = b_packed & 0x0F
        high = b_packed >> 4
        is_high = (offs_k & 1)[:, None]
        b_nibble = tl.where(is_high == 1, high, low)

        base = tl.load(LUT_ptr + b_nibble)

        scale_idx = offs_k // GROUP_SIZE
        scale = tl.load(
            S_ptr + scale_idx[:, None] * stride_sk + offs_n[None, :] * stride_sn,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0,
        ).to(tl.float16)
        b = base * scale * scale2

        acc += tl.dot(a, b)
        k += BLOCK_K

    c = acc.to(tl.float16)
    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def nvfp4_gemm_dual_kernel(
    A_ptr,
    B1_ptr,
    S1_ptr,
    S1_2_ptr,
    B2_ptr,
    S2_ptr,
    S2_2_ptr,
    LUT_ptr,
    C1_ptr,
    C2_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_b1n,
    stride_b1k,
    stride_s1n,
    stride_s1k,
    stride_b2n,
    stride_b2k,
    stride_s2n,
    stride_s2k,
    stride_c1m,
    stride_c1n,
    stride_c2m,
    stride_c2n,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    scale1_2 = tl.load(S1_2_ptr).to(tl.float16)
    scale2_2 = tl.load(S2_2_ptr).to(tl.float16)

    k = 0
    while k < K:
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0,
        ).to(tl.float16)

        packed_idx = offs_k >> 1
        b1_packed = tl.load(
            B1_ptr + packed_idx[:, None] * stride_b1k + offs_n[None, :] * stride_b1n,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0,
        )
        low1 = b1_packed & 0x0F
        high1 = b1_packed >> 4
        is_high = (offs_k & 1)[:, None]
        b1_nibble = tl.where(is_high == 1, high1, low1)
        base1 = tl.load(LUT_ptr + b1_nibble)
        scale_idx = offs_k // GROUP_SIZE
        scale1 = tl.load(
            S1_ptr + scale_idx[:, None] * stride_s1k + offs_n[None, :] * stride_s1n,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0,
        ).to(tl.float16)
        b1 = base1 * scale1 * scale1_2

        b2_packed = tl.load(
            B2_ptr + packed_idx[:, None] * stride_b2k + offs_n[None, :] * stride_b2n,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0,
        )
        low2 = b2_packed & 0x0F
        high2 = b2_packed >> 4
        b2_nibble = tl.where(is_high == 1, high2, low2)
        base2 = tl.load(LUT_ptr + b2_nibble)
        scale2 = tl.load(
            S2_ptr + scale_idx[:, None] * stride_s2k + offs_n[None, :] * stride_s2n,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0,
        ).to(tl.float16)
        b2 = base2 * scale2 * scale2_2

        acc1 += tl.dot(a, b1)
        acc2 += tl.dot(a, b2)
        k += BLOCK_K

    c1 = acc1.to(tl.float16)
    c2 = acc2.to(tl.float16)
    tl.store(
        C1_ptr + offs_m[:, None] * stride_c1m + offs_n[None, :] * stride_c1n,
        c1,
        mask=mask_m[:, None] & mask_n[None, :],
    )
    tl.store(
        C2_ptr + offs_m[:, None] * stride_c2m + offs_n[None, :] * stride_c2n,
        c2,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def nvfp4_gemm_kernel_fp8(
    A_ptr,
    B_ptr,
    S_ptr,
    S2_ptr,
    LUT_ptr,
    B_COL_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_sn,
    stride_sk,
    stride_cm,
    stride_cn,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_A: tl.constexpr,
    SCALE_B: tl.constexpr,
    SCALE_BCOL: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    scale2 = tl.load(S2_ptr).to(tl.float16)

    k = 0
    while k < K:
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0,
        ).to(tl.float16)

        packed_idx = offs_k >> 1
        b_packed = tl.load(
            B_ptr + packed_idx[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0,
        )
        low = b_packed & 0x0F
        high = b_packed >> 4
        is_high = (offs_k & 1)[:, None]
        b_nibble = tl.where(is_high == 1, high, low)

        base = tl.load(LUT_ptr + b_nibble)

        scale_idx = offs_k // GROUP_SIZE
        scale = tl.load(
            S_ptr + scale_idx[:, None] * stride_sk + offs_n[None, :] * stride_sn,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0,
        ).to(tl.float16)
        b = base * scale * scale2

        if SCALE_BCOL:
            bcol = tl.load(B_COL_ptr + offs_n, mask=mask_n, other=1.0).to(tl.float32)
            bcol = tl.where(bcol == 0, 1.0, bcol)
            b_fp8 = (b / bcol[None, :]).to(_FP8_DTYPE)
        elif SCALE_B:
            bmax = tl.max(tl.abs(b), axis=0).to(tl.float32)
            bmax = tl.where(bmax == 0, 1.0, bmax)
            b_fp8 = (b / bmax[None, :]).to(_FP8_DTYPE)
        else:
            b_fp8 = b.to(_FP8_DTYPE)
        if SCALE_A:
            amax = tl.max(tl.abs(a), axis=1).to(tl.float32)
            amax = tl.where(amax == 0, 1.0, amax)
            a_fp8 = (a / amax[:, None]).to(_FP8_DTYPE)
            if SCALE_BCOL:
                acc += tl.dot(a_fp8, b_fp8) * (amax[:, None] * bcol[None, :])
            elif SCALE_B:
                acc += tl.dot(a_fp8, b_fp8) * (amax[:, None] * bmax[None, :])
            else:
                acc += tl.dot(a_fp8, b_fp8) * amax[:, None]
        else:
            a_fp8 = a.to(_FP8_DTYPE)
            if SCALE_BCOL:
                acc += tl.dot(a_fp8, b_fp8) * bcol[None, :]
            elif SCALE_B:
                acc += tl.dot(a_fp8, b_fp8) * bmax[None, :]
            else:
                acc += tl.dot(a_fp8, b_fp8)
        k += BLOCK_K

    c = acc.to(tl.float16)
    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def nvfp4_gemm_gather_kernel(
    A_ptr,
    B_ptr,
    S_ptr,
    S2_ptr,
    LUT_ptr,
    E_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_se,
    stride_sn,
    stride_sk,
    stride_s2e,
    stride_cm,
    stride_cn,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    exp_ids = tl.load(E_ptr + offs_m, mask=mask_m, other=0).to(tl.int32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    scale2 = tl.load(S2_ptr + exp_ids * stride_s2e, mask=mask_m, other=1.0).to(tl.float16)

    k = 0
    while k < K:
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0,
        ).to(tl.float16)

        packed_idx = offs_k >> 1
        b_ptrs = (
            B_ptr
            + exp_ids[:, None, None] * stride_be
            + offs_n[None, None, :] * stride_bn
            + packed_idx[None, :, None] * stride_bk
        )
        b_packed = tl.load(
            b_ptrs,
            mask=mask_m[:, None, None] & mask_k[None, :, None] & mask_n[None, None, :],
            other=0,
        )
        low = b_packed & 0x0F
        high = b_packed >> 4
        is_high = (offs_k & 1)[None, :, None]
        b_nibble = tl.where(is_high == 1, high, low)

        base = tl.load(LUT_ptr + b_nibble)

        scale_idx = offs_k // GROUP_SIZE
        s_ptrs = (
            S_ptr
            + exp_ids[:, None, None] * stride_se
            + offs_n[None, None, :] * stride_sn
            + scale_idx[None, :, None] * stride_sk
        )
        scale = tl.load(
            s_ptrs,
            mask=mask_m[:, None, None] & mask_k[None, :, None] & mask_n[None, None, :],
            other=0,
        ).to(tl.float16)

        b = base * scale * scale2[:, None, None]
        acc += tl.sum(a[:, :, None] * b, axis=1)
        k += BLOCK_K

    c = acc.to(tl.float16)
    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def nvfp4_linear(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
    w_scale_2: torch.Tensor,
    w_scale_col: torch.Tensor | None,
    bias: torch.Tensor | None,
    group_size: int,
) -> torch.Tensor:
    if not x.is_cuda:
        raise RuntimeError("nvfp4_linear requires CUDA tensors")
    out_dtype = x.dtype
    if x.dtype not in (torch.float16, torch.bfloat16):
        x = x.to(torch.float16)
    x_2d = x.reshape(-1, x.shape[-1])
    M, K = x_2d.shape
    N = w_packed.shape[0]
    if w_packed.shape[1] * 2 != K:
        raise ValueError("Packed weight shape does not match input features")
    c = torch.empty((M, N), device=x.device, dtype=torch.float16)

    lut = _fp4_lut_on_device(x.device)
    def _select_config(m, n, k):
        # Decode: small M; Prefill: larger M. (Swept on RTX 4090 Oct 2026:
        # in-model best is the stock config; microbench wins don't transfer.)
        if m <= 16:
            return dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=3)
        if m <= 64:
            if k >= 8192:
                return dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=128, num_warps=8, num_stages=4)
            return dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=4)
        if k >= 8192:
            return dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, num_warps=8, num_stages=4)
        return dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_warps=8, num_stages=4)

    cfg = _select_config(M, N, K)
    grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]),)
    if _FP8_ENABLED:
        if not _fp8_supported():
            raise RuntimeError("NVFP4_FP8 is enabled but FP8 is not supported on this device.")
        try:
            use_bcol = _FP8_BCOL and w_scale_col is not None and w_scale_col.numel() == N
            if not use_bcol:
                w_scale_col = w_scale_2
            nvfp4_gemm_kernel_fp8[grid](
                x_2d,
                w_packed,
                w_scale,
                w_scale_2,
                lut,
                w_scale_col,
                c,
                M,
                N,
                K,
                x_2d.stride(0),
                x_2d.stride(1),
                w_packed.stride(0),
                w_packed.stride(1),
                w_scale.stride(0),
                w_scale.stride(1),
                c.stride(0),
                c.stride(1),
                GROUP_SIZE=group_size,
                BLOCK_M=cfg["BLOCK_M"],
                BLOCK_N=cfg["BLOCK_N"],
                BLOCK_K=cfg["BLOCK_K"],
                SCALE_A=_FP8_A_SCALE,
                SCALE_B=_FP8_B_SCALE,
                SCALE_BCOL=use_bcol,
                num_warps=cfg["num_warps"],
                num_stages=cfg["num_stages"],
            )
        except Exception as exc:
            raise RuntimeError(
                "NVFP4_FP8 kernel failed. Set NVFP4_FP8=0 to use the FP16 GEMM path."
            ) from exc
    else:
        nvfp4_gemm_kernel[grid](
            x_2d,
            w_packed,
            w_scale,
            w_scale_2,
            lut,
            c,
            M,
            N,
            K,
            x_2d.stride(0),
            x_2d.stride(1),
            w_packed.stride(0),
            w_packed.stride(1),
            w_scale.stride(0),
            w_scale.stride(1),
            c.stride(0),
            c.stride(1),
            GROUP_SIZE=group_size,
            BLOCK_M=cfg["BLOCK_M"],
            BLOCK_N=cfg["BLOCK_N"],
            BLOCK_K=cfg["BLOCK_K"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )
    if bias is not None:
        c = c + bias.to(c.dtype)
    return c.to(out_dtype).view(*x.shape[:-1], N)


def nvfp4_linear_dual(
    x: torch.Tensor,
    w1_packed: torch.Tensor,
    w1_scale: torch.Tensor,
    w1_scale_2: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_scale_2: torch.Tensor,
    bias1: torch.Tensor | None,
    bias2: torch.Tensor | None,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _FP8_ENABLED:
        y1 = nvfp4_linear(x, w1_packed, w1_scale, w1_scale_2, None, bias1, group_size)
        y2 = nvfp4_linear(x, w2_packed, w2_scale, w2_scale_2, None, bias2, group_size)
        return y1, y2
    if not x.is_cuda:
        raise RuntimeError("nvfp4_linear_dual requires CUDA tensors")
    out_dtype = x.dtype
    if x.dtype not in (torch.float16, torch.bfloat16):
        x = x.to(torch.float16)
    x_2d = x.reshape(-1, x.shape[-1])
    M, K = x_2d.shape
    N1 = w1_packed.shape[0]
    N2 = w2_packed.shape[0]
    if N1 != N2:
        raise ValueError("Dual NVFP4 expects matching output sizes")
    N = N1
    if w1_packed.shape[1] * 2 != K or w2_packed.shape[1] * 2 != K:
        raise ValueError("Packed weight shape does not match input features")
    c1 = torch.empty((M, N), device=x.device, dtype=torch.float16)
    c2 = torch.empty((M, N), device=x.device, dtype=torch.float16)

    lut = _fp4_lut_on_device(x.device)

    def _select_config(m, n, k):
        if m <= 16:
            return dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=3)
        if m <= 64:
            if k >= 8192:
                return dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=128, num_warps=8, num_stages=4)
            return dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=4)
        if k >= 8192:
            return dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, num_warps=8, num_stages=4)
        return dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_warps=8, num_stages=4)

    cfg = _select_config(M, N, K)
    grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]),)
    nvfp4_gemm_dual_kernel[grid](
        x_2d,
        w1_packed,
        w1_scale,
        w1_scale_2,
        w2_packed,
        w2_scale,
        w2_scale_2,
        lut,
        c1,
        c2,
        M,
        N,
        K,
        x_2d.stride(0),
        x_2d.stride(1),
        w1_packed.stride(0),
        w1_packed.stride(1),
        w1_scale.stride(0),
        w1_scale.stride(1),
        w2_packed.stride(0),
        w2_packed.stride(1),
        w2_scale.stride(0),
        w2_scale.stride(1),
        c1.stride(0),
        c1.stride(1),
        c2.stride(0),
        c2.stride(1),
        GROUP_SIZE=group_size,
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_N=cfg["BLOCK_N"],
        BLOCK_K=cfg["BLOCK_K"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    if bias1 is not None:
        c1 = c1 + bias1.to(c1.dtype)
    if bias2 is not None:
        c2 = c2 + bias2.to(c2.dtype)
    return c1.to(out_dtype).view(*x.shape[:-1], N), c2.to(out_dtype).view(*x.shape[:-1], N)


def nvfp4_linear_gather(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
    w_scale_2: torch.Tensor,
    expert_ids: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    if not x.is_cuda:
        raise RuntimeError("nvfp4_linear_gather requires CUDA tensors")
    if x.dtype not in (torch.float16, torch.bfloat16):
        x = x.to(torch.float16)
    x_2d = x.reshape(-1, x.shape[-1])
    M, K = x_2d.shape
    N = w_packed.shape[1]
    if w_packed.shape[2] * 2 != K:
        raise ValueError("Packed weight shape does not match input features")
    if expert_ids.numel() != M:
        raise ValueError("expert_ids must have one entry per row in x")
    c = torch.empty((M, N), device=x.device, dtype=torch.float16)

    lut = _fp4_lut_on_device(x.device)
    expert_ids = expert_ids.to(torch.int32)
    if w_scale_2 is None:
        w_scale_2 = torch.ones((w_packed.shape[0],), device=x.device, dtype=torch.float32)

    cfg = dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=2)
    grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]),)
    nvfp4_gemm_gather_kernel[grid](
        x_2d,
        w_packed,
        w_scale,
        w_scale_2,
        lut,
        expert_ids,
        c,
        M,
        N,
        K,
        x_2d.stride(0),
        x_2d.stride(1),
        w_packed.stride(0),
        w_packed.stride(1),
        w_packed.stride(2),
        w_scale.stride(0),
        w_scale.stride(1),
        w_scale.stride(2),
        w_scale_2.stride(0),
        c.stride(0),
        c.stride(1),
        GROUP_SIZE=group_size,
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_N=cfg["BLOCK_N"],
        BLOCK_K=cfg["BLOCK_K"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    return c.to(x.dtype).view(*x.shape[:-1], N)
