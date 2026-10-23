"""Marlin NVFP4 GEMM correctness test — GPU-optional.

Run later on an idle RTX 4090 (e.g. after the prod reload finishes):

    pytest tests/test_marlin_correctness.py -v

On CPU (or without vLLM 0.32) every test skips — safe to collect anywhere.

Reference: quantize a random fp16 weight to NVFP4 (e4m3 group-16 scales),
repack with the vendored repack_linear, run the Marlin GEMM via
marlin_linear(), and compare against a plain torch dequant-reference
matmul. Tolerance reflects fp16 Marlin accumulation (~1e-2 relative).
"""

from __future__ import annotations

import math
import os

import pytest

torch = pytest.importorskip("torch")

# ---------------------------------------------------------------- skip gates
try:
    from vllm_adallm_kernels import marlin_vendored  # noqa: F401

    VLLM_OK = marlin_vendored._IMPORT_OK
except Exception:
    marlin_vendored = None
    VLLM_OK = False

CUDA_OK = torch.cuda.is_available()
ADA_OK = CUDA_OK and torch.cuda.get_device_capability(0)[0] * 10 + torch.cuda.get_device_capability(0)[1] == 89

requires_gpu = pytest.mark.skipif(
    not (VLLM_OK and CUDA_OK), reason="needs CUDA + vLLM 0.32 with marlin utils"
)


# ---------------------------------------------------------------- helpers
_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float16,
)


def _quantize_nvfp4(w: torch.Tensor, group: int = 16):
    """Reference NVFP4 quantizer matching adallm's convention:
    dequant = e2m1(q) * s_per_group * s2_global, s2 = 1.0."""
    N, K = w.shape
    wg = w.float().view(N, K // group, group)
    amax = wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    s = (amax / 6.0).clamp(max=448.0)  # e4m3-representable group scale
    q = (wg / s).clamp(-6.0, 6.0)
    # round to nearest e2m1 code
    codes = torch.argmin((q.unsqueeze(-1) - _E2M1_LUT.float()).abs(), dim=-1)
    deq = _E2M1_LUT.float()[codes]
    wq_packed = (
        codes[..., 0::2].to(torch.uint8) | (codes[..., 1::2].to(torch.uint8) << 4)
    ).reshape(N, K // 2)
    return wq_packed, s.reshape(N, K // group)


class _FakeLinear(torch.nn.Module):
    """Mimics adallm LinearBase fields consumed by repack_linear()."""

    def __init__(self, wq, s, s2, group):
        super().__init__()
        self.weight = torch.nn.Parameter(wq, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(s.to(torch.float16), requires_grad=False)
        self.weight_scale_2 = torch.nn.Parameter(
            torch.tensor([s2], dtype=torch.float32), requires_grad=False
        )
        self.nvfp4_group_size = group


def _reference_matmul(x, wq, s, group, s2=1.0):
    N, K_half = wq.shape
    K = K_half * 2
    lo = (wq & 0x0F).long()
    hi = (wq >> 4).long()
    idx = torch.stack([lo, hi], dim=-1).reshape(N, K)
    base = _E2M1_LUT[idx]
    scale = s.float().repeat_interleave(group, dim=1)
    w = base * scale * s2
    return (x.float() @ w.T.float())


# ---------------------------------------------------------------- tests
def test_cpu_reference_selfconsistent():
    """CPU-only: our reference quantizer round-trips within fp4 error."""
    torch.manual_seed(0)
    w = torch.randn(64, 256, dtype=torch.float16)
    wq, s = _quantize_nvfp4(w)
    ref = _reference_matmul(w[:1], wq, s, 16)
    rel = (ref - (w[:1].float() @ w.float().T)).abs().max() / ref.abs().max()
    assert rel < 0.25  # fp4 quantization noise


@requires_gpu
def test_marlin_matches_dequant_reference():
    """GPU: vendored repack + marlin_linear ≈ dequant reference matmul."""
    torch.manual_seed(0)
    dev = torch.device("cuda")
    N, K, M = 4096, 4096, 4  # N >= _MARLIN_N_MIN default 2048
    w = torch.randn(N, K, dtype=torch.float16, device=dev) * 0.05
    wq, s = _quantize_nvfp4(w.cpu())
    layer = _FakeLinear(wq.cuda(), s.cuda(), 1.0, 16).to(dev)

    marlin_vendored.repack_linear(layer)
    assert getattr(layer, "marlin_ready", False)

    x = torch.randn(M, K, dtype=torch.float16, device=dev)
    out = marlin_vendored.marlin_linear(layer, x, None)
    assert out is not None and out.shape == (M, N)

    ref = _reference_matmul(x.cpu(), wq, s, 16).to(dev)
    rel = (out.float() - ref).abs().max() / ref.abs().max()
    assert rel < 2e-2, f"marlin vs dequant reference rel err {rel:.3e}"


@requires_gpu
def test_marlin_scale_factor_guard():
    """GPU: the adallm scale fix — scales ~2e-3 must survive the S0E5M3
    conversion (stock path would zero them all)."""
    torch.manual_seed(0)
    dev = torch.device("cuda")
    N, K = 2048, 2048
    w = torch.randn(N, K, dtype=torch.float16, device=dev) * 0.01
    wq, s = _quantize_nvfp4(w.cpu())
    # force small scales like the real checkpoint (~2e-3)
    s = s * 1e-3
    layer = _FakeLinear(wq.cuda(), s.cuda(), 1.0, 16).to(dev)
    marlin_vendored.repack_linear(layer)
    ws = layer.weight_scale.view(torch.float8_e4m3fn)
    # after repack, weight_scale is S0E5M3-packed; the guard is that the
    # GEMM still matches the reference (zeroed scales => all-zero output)
    x = torch.randn(2, K, dtype=torch.float16, device=dev)
    out = marlin_vendored.marlin_linear(layer, x, None)
    ref = _reference_matmul(x.cpu(), wq, s, 16).to(dev)
    assert out.float().abs().max() > 0, "scales were zeroed (scale_factor bug)"
    rel = (out.float() - ref).abs().max() / ref.abs().max()
    assert rel < 2e-2
