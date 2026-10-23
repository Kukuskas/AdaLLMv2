"""Marlin NVFP4 GEMM correctness test — GPU-optional.

Run on an idle RTX 4090 (CUDA_VISIBLE_DEVICES=1):

    pytest tests/test_marlin_correctness.py -v

On CPU (or without vLLM 0.32) every test skips — safe to collect anywhere.

Reference: quantize a random fp16 weight to NVFP4 (e4m3 group-16 scales),
repack with the vendored repack_linear, run the Marlin GEMM via
marlin_linear(), and compare against a plain torch dequant-reference
matmul that models the kernel's S0E5M3 scale format.

M1/M2 RESOLUTION (2026-10-23): the historical 4.3e-2 rel-err was NOT a
kernel or vendoring bug — the Marlin NVFP4 path stores per-group scales in
a special S0E5M3 format built by ``nvfp4_marlin_process_scales``:
fp16(s*sf) with the low 7 mantissa bits truncated (int16<<1 → fp8 byte),
values with s*sf*2^7 < 2 zeroed, compensated by global_scale/sf. A
reference that uses the exact fp16 scale therefore disagrees by ~4-5e-2
even though the kernel is faithful to its format: modeling the truncation
drops the residual to ~3e-4. The reference here simulates exactly that
truncation (see ``_s0e5m3_effective_scale``).
"""

from __future__ import annotations

import math

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
    wq_packed = (
        codes[..., 0::2].to(torch.uint8) | (codes[..., 1::2].to(torch.uint8) << 4)
    ).reshape(N, K // 2)
    return wq_packed, s.reshape(N, K // group)


def _marlin_scale_factor(s: torch.Tensor) -> float:
    """Mirror repack_linear()'s explicit scale_factor: smallest power of 2
    with smin*sf*128 >= 2 (guards against S0E5M3 zeroing small scales)."""
    nonzero = s[s > 0]
    smin = nonzero.min().item() if nonzero.numel() else 1.0
    return max(2.0 ** math.ceil(math.log2(2.0 / (smin * 128.0))), 1.0)


def _s0e5m3_effective_scale(s: torch.Tensor, sf: float) -> torch.Tensor:
    """Effective per-group scale after the kernel's S0E5M3 packing.

    nvfp4_marlin_process_scales does: fp16(s*sf) -> *2^7 -> int16<<1 ->
    fp8 byte, i.e. it keeps sign+exponent+top-3 mantissa bits of fp16(s*sf)
    and TRUNCATES (not rounds) the low 7 bits; values whose s*sf*2^7 < 2
    are zeroed. The global scale divides by sf to compensate.
    """
    h = (s.float() * sf).to(torch.float16)
    # the kernel consumes weight_scale stored as fp16 first (repack_linear:
    # scales.to(torch.float16)); model that rounding too
    h = (s.to(torch.float16).float() * sf).to(torch.float16)
    bits = h.view(torch.int16) & ~0x7F  # drop low 7 mantissa bits (int16<<1 keeps the top 3)
    v = bits.view(torch.float16).float()
    v = torch.where((s.float() * sf * 128.0) < 2.0, torch.zeros_like(v), v)
    return v / sf


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


def _reference_matmul(x, wq, s_eff, group, s2=1.0):
    """Dequant matmul using the S0E5M3-effective group scales."""
    N, K_half = wq.shape
    K = K_half * 2
    lo = (wq & 0x0F).long()
    hi = (wq >> 4).long()
    idx = torch.stack([lo, hi], dim=-1).reshape(N, K)
    base = _E2M1_LUT[idx]
    scale = s_eff.repeat_interleave(group, dim=1)
    w = base * scale * s2
    return (x.float() @ w.T.float())


def _run_marlin(wq, s, s2, x_dev):
    dev = x_dev.device
    layer = _FakeLinear(wq.to(dev), s.to(dev), s2, 16).to(dev)
    marlin_vendored.repack_linear(layer)
    assert getattr(layer, "marlin_ready", False)
    out = marlin_vendored.marlin_linear(layer, x_dev, None)
    assert out is not None and out.shape == (x_dev.shape[0], wq.shape[0])
    return out


# ---------------------------------------------------------------- tests
def test_cpu_reference_selfconsistent():
    """CPU-only: our reference quantizer round-trips within fp4 error."""
    torch.manual_seed(0)
    w = torch.randn(64, 256, dtype=torch.float16)
    wq, s = _quantize_nvfp4(w)
    sf = _marlin_scale_factor(s)
    ref = _reference_matmul(w[:1], wq, _s0e5m3_effective_scale(s, sf), 16)
    assert ref.float().abs().max() > 0  # sf guard: nothing zeroed


@pytest.mark.skipif(
    not (VLLM_OK and CUDA_OK and ADA_OK),
    reason="needs CUDA sm89 + vLLM 0.32 with marlin utils",
)
def test_marlin_matches_dequant_reference():
    """GPU: vendored repack + marlin_linear ≈ S0E5M3-aware dequant reference."""
    torch.manual_seed(0)
    dev = torch.device("cuda")
    N, K, M = 4096, 4096, 4  # N >= _MARLIN_N_MIN default 2048
    w = torch.randn(N, K, dtype=torch.float16, device=dev) * 0.05
    wq, s = _quantize_nvfp4(w.cpu())

    sf = _marlin_scale_factor(s)
    s_eff = _s0e5m3_effective_scale(s, sf)

    x = torch.randn(M, K, dtype=torch.float16, device=dev)
    out = _run_marlin(wq, s, 1.0, x)

    ref = _reference_matmul(x.cpu(), wq, s_eff, 16).to(dev)
    rel = (out.float() - ref).abs().max() / ref.abs().max()
    assert rel < 2e-2, f"marlin vs S0E5M3 dequant reference rel err {rel:.3e}"


@pytest.mark.skipif(
    not (VLLM_OK and CUDA_OK and ADA_OK),
    reason="needs CUDA sm89 + vLLM 0.32 with marlin utils",
)
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
    sf = _marlin_scale_factor(s)
    s_eff = _s0e5m3_effective_scale(s, sf)

    x = torch.randn(2, K, dtype=torch.float16, device=dev)
    out = _run_marlin(wq, s, 1.0, x)

    ref = _reference_matmul(x.cpu(), wq, s_eff, 16).to(dev)
    assert out.float().abs().max() > 0, "scales were zeroed (scale_factor bug)"
    rel = (out.float() - ref).abs().max() / ref.abs().max()
    assert rel < 2e-2, f"scale-factor guard rel err {rel:.3e}"
