"""Marlin NVFP4 kernel integration (vLLM kernels, on-the-fly repack).

Enabled with ADALLM_MARLIN=1. Repacks each LinearBase nvfp4 weight into
Marlin layout after load and dispatches small-M GEMMs to
apply_fp4_marlin_linear (fp16 activations).

Scale path (root-caused 2026-10-21, see shared findings):
  - _nvfp4_compute_scale_factor() returns 1.0 for fp16 activations
  - nvfp4_marlin_process_scales() zeroes any scale < 2 after *2^7
  - our per-group scales (~2e-3..2e-2) would all be zeroed
  - FIX: explicit scale_factor so smin*sf*128 >= 2, global scale /sf.
    Our quantizer convention: dequant = q4 * s (group=16), global = 1.0
    (weight_scale_2 folded in at repack time when present).
"""
import math
import os
import torch
from torch import nn

_MARLIN_ENV = os.getenv("ADALLM_MARLIN", "0").strip().lower()
MARLIN_ENABLED = _MARLIN_ENV not in ("0", "false", "off", "")
# decode-batch GEMMs at or below this M go to Marlin; above -> Triton prefill kernel
_MARLIN_M_MAX = int(os.getenv("ADALLM_MARLIN_M_MAX", "8"))
# Marlin loses below this N (kv_proj 1024, in_proj_a/b 48) — keep Triton there
# (subagent sweep 2026-10-21: marlin 1.1-1.7x slower on tiny-N shapes)
_MARLIN_N_MIN = int(os.getenv("ADALLM_MARLIN_N_MIN", "2048"))

# Round2-D tiny-N fast path (ADALLM_TINYGEMM=1): layers with N <= threshold
# get their NVFP4 weights DEQUANTIZED once at load into a plain fp16/bf16
# weight and run through cuBLAS F.linear. At M=1 the Triton kernel is
# launch/occupancy-bound (57-63 us); in-graph F.linear on a dequantized
# weight is 3.5-12.6 us (20x). Extra VRAM ~0.4-0.9 GB.
_TINYGEMM_ENV = os.getenv("ADALLM_TINYGEMM", "0").strip().lower()
TINYGEMM_ENABLED = _TINYGEMM_ENV not in ("0", "false", "off", "")
_TINY_N_MAX = int(os.getenv("ADALLM_TINYGEMM_N_MAX", "1024"))
# fuse in_proj_a+b (2x48 -> 96) and k+v_proj (2x1024 -> 2048) into single
# F.linear calls (weights concatenated). Default on; disable to isolate.
_TINYGEMM_FUSE = os.getenv("ADALLM_TINYGEMM_FUSE", "1").strip().lower() not in ("0", "false", "off")

# Round4-B measured per-shape dispatch table (default OFF). Comma-separated
# "NxK" list of NVFP4 GEMM shapes to route to cuBLAS on a dequantized fp16
# weight instead of Marlin, regardless of N. Measured per-shape at decode M
# (see probes/round4-b/bench_dispatch.py + findings round4-b-dispatch):
# env-gated static table; modules listed here are dequantized BEFORE the
# Marlin repack so the packed copy is the one freed.
_DEQ_SHAPES_RAW = os.getenv("ADALLM_DEQ_SHAPES", "").strip()
DEQ_SHAPES = set()
for _tok in _DEQ_SHAPES_RAW.split(","):
    _tok = _tok.strip().lower()
    if not _tok:
        continue
    try:
        _n, _k = _tok.split("x")
        DEQ_SHAPES.add((int(_n), int(_k)))
    except ValueError:
        print(f"[deq-shapes] ignoring malformed token {_tok!r}")

try:
    from vllm.model_executor.kernels.linear.nvfp4.marlin import (
        apply_fp4_marlin_linear,
        is_fp4_marlin_supported,
    )
    from vllm import _custom_ops as _ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
        marlin_padded_nk,
        marlin_pad_qweight,
        marlin_pad_scales,
        marlin_permute_scales,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        nvfp4_marlin_process_scales,
        nvfp4_marlin_process_global_scale,
    )
    _IMPORT_OK = True
except Exception as e:  # vllm not installed / older layout
    _IMPORT_OK = False
    _IMPORT_ERR = e


def marlin_available() -> bool:
    return _IMPORT_OK and MARLIN_ENABLED and is_fp4_marlin_supported()


def repack_linear(module: nn.Module) -> None:
    """Convert one LinearBase (nvfp4 kernel mode) to Marlin weights in-place.

    Keeps the original uint8 weight as module.weight_triton for fallback.
    """
    assert _IMPORT_OK, f"marlin imports failed: {_IMPORT_ERR}"
    weight = module.weight  # uint8 [N, K/2]
    scales = module.weight_scale  # fp16 [N, K/16] (values e4m3-representable)
    scale2 = module.weight_scale_2  # scalar tensor (1.0 unless checkpoint had one)
    group_size = module.nvfp4_group_size
    N, K_half = weight.shape
    K = K_half * 2
    dev = weight.device

    # fold tensor-level scale_2 into per-group scales (our kernel multiplies
    # both; Marlin takes a single per-group scale + global)
    s2 = float(scale2.reshape(-1)[0].item()) if scale2 is not None and scale2.numel() else 1.0
    scales_f = scales.float() * s2

    padded_n, padded_k = marlin_padded_nk(N, K, group_size)
    perm = torch.empty(0, dtype=torch.int, device=dev)
    qweight = weight.detach().view(torch.int32).T.contiguous()
    qweight = marlin_pad_qweight(qweight, N, K, padded_n, padded_k)
    marlin_qweight = _ops.gptq_marlin_repack(
        b_q_weight=qweight, perm=perm, size_k=padded_k, size_n=padded_n,
        num_bits=4, is_a_8bit=False,
    )

    ws = scales_f.T.contiguous().to(torch.float16)
    ws = marlin_pad_scales(ws, N, K, padded_n, padded_k, group_size)
    ws = marlin_permute_scales(s=ws, size_k=padded_k, size_n=padded_n,
                               group_size=group_size, is_a_8bit=False)

    nonzero = scales_f[scales_f > 0]
    smin = nonzero.min().item() if nonzero.numel() else 1.0
    sf = max(2.0 ** math.ceil(math.log2(2.0 / (smin * 128.0))), 1.0)
    ws_p, sf_ret = nvfp4_marlin_process_scales(ws, scale_factor=sf, a_dtype=torch.float16)
    gs = nvfp4_marlin_process_global_scale(
        torch.tensor([1.0], device=dev, dtype=torch.float32), torch.float16
    ) / sf_ret

    module.weight_triton = module.weight  # kept only if not freeing
    module.weight_scale_triton = module.weight_scale
    if os.getenv("ADALLM_MARLIN_FREE_TRITON", "1").strip().lower() not in ("0", "false", "off"):
        # memory: model is ~20GB packed; keeping both copies doubles it -> OOM.
        # Marlin handles any M (prefill too), so drop the Triton copy.
        module.weight_triton = None
        module.weight_scale_triton = None
    module.weight = nn.Parameter(marlin_qweight, requires_grad=False)
    module.weight_scale = nn.Parameter(ws_p, requires_grad=False)
    module.weight_marlin_global = gs.reshape(1)
    module.workspace = marlin_make_workspace_new(dev)
    module.marlin_ready = True
    module.marlin_n = N
    module.marlin_k = K
    module.marlin_group_size = group_size


def repack_model(model: nn.Module) -> int:
    """Repack every nvfp4 LinearBase in the model. Returns count."""
    if marlin_available():
        count = _repack_model_impl(model)
    elif TINYGEMM_ENABLED or DEQ_SHAPES:
        # no Marlin: nothing to repack, only dequant paths run
        count = 0
    else:
        return 0
    if TINYGEMM_ENABLED or DEQ_SHAPES:
        _dequant_tiny_layers(model)
    return count


def _dequant_tiny(module: nn.Module) -> None:
    """Dequantize one tiny-N LinearBase's NVFP4 weight into a plain dtype
    tensor (module.weight_deq) for cuBLAS F.linear dispatch."""
    weight = module.weight  # uint8 [N, K/2]
    scales = module.weight_scale  # fp16 [N, K/16]
    scale2 = module.weight_scale_2  # scalar fp32
    group = module.nvfp4_group_size
    N, K_half = weight.shape
    K = K_half * 2
    # match activation dtype: model default dtype governs (bf16 normally).
    # NVFP4_DEQ_DTYPE=fp16 forces fp16 weights (closer to the Triton kernel's
    # fp16 math; bf16 dequant adds 2^-9 relative rounding that flips
    # near-tie greedy tokens).
    _deq_dt = os.getenv("NVFP4_DEQ_DTYPE", "model").strip().lower()
    if _deq_dt == "fp16":
        act_dtype = torch.float16
    else:
        act_dtype = torch.get_default_dtype()
    from adallm.ops.nvfp4_gemm import _fp4_lut_on_device
    lut = _fp4_lut_on_device(weight.device)
    lo = (weight & 0x0F).long()
    hi = (weight >> 4).long()
    idx = torch.stack([lo, hi], dim=-1).reshape(N, K)  # even k = lo nibble
    base = lut[idx]  # fp16 [N, K]
    scale = scales.float().repeat_interleave(group, dim=1)  # [N, K]
    s2 = float(scale2.reshape(-1)[0].item()) if scale2 is not None and scale2.numel() else 1.0
    w = base.float() * scale * s2
    module.weight_deq = w.to(act_dtype).contiguous()
    module._deq_n = N
    # free the packed copy only when Marlin is not using it (tiny layers are
    # never repacked to Marlin, so the packed tensor is dead weight)
    module.weight = nn.Parameter(torch.empty(0, dtype=torch.uint8), requires_grad=False)
    module.weight_scale = nn.Parameter(torch.empty(0, dtype=torch.float16), requires_grad=False)
    module.weight_scale_2 = nn.Parameter(torch.ones([], dtype=torch.float32), requires_grad=False)
    module.tinyygemm_ready = True


def _dequant_tiny_layers(model: nn.Module) -> int:
    """Dequant all tiny-N nvfp4 LinearBase layers (and optionally fuse
    in_proj_a+b / k+v pairs by concatenating dequantized weights)."""
    count = 0
    for name, module in model.named_modules():
        if not getattr(module, "nvfp4_enabled", False):
            continue
        if getattr(module, "nvfp4_mode", None) != "kernel":
            continue
        w = getattr(module, "weight", None)
        if w is None or w.dtype != torch.uint8 or w.ndim != 2 or w.numel() == 0:
            continue
        if w.shape[0] > _TINY_N_MAX and (w.shape[0], w.shape[1] * 2) not in DEQ_SHAPES:
            continue
        if getattr(module, "tinyygemm_ready", False):
            continue
        if getattr(module, "marlin_ready", False):
            continue  # marlin already took it
        try:
            _dequant_tiny(module)
            count += 1
        except Exception as e:
            print(f"[tinyygemm] dequant failed for {name}: {type(e).__name__}: {e}")
    # pairwise fusion (a+b, k+v): set both modules to use the fused weight
    if _TINYGEMM_FUSE:
        count_f = _fuse_pairs(model)
        count += count_f
    print(f"[tinyygemm] dequantized {count} tiny-N layers")
    return count


_PAIR_ATTRS = {
    # (parent module attr, child A attr, child B attr)
    "in_proj_a": ("in_proj_b",),
    "k_proj": ("v_proj",),
}


def _fuse_pairs(model: nn.Module) -> int:
    """Concatenate dequantized weights of (in_proj_a,in_proj_b) and
    (k_proj,v_proj) sibling pairs so each pair runs ONE F.linear."""
    fused = 0
    for name, module in model.named_modules():
        for attr_a, sibs in _PAIR_ATTRS.items():
            if not hasattr(module, attr_a):
                continue
            ma = getattr(module, attr_a)
            mb = getattr(module, sibs[0])
            if ma is mb:
                continue
            wa = getattr(ma, "weight_deq", None)
            wb = getattr(mb, "weight_deq", None)
            if wa is None or wb is None:
                continue
            if getattr(ma, "fused_pair", False) or getattr(mb, "fused_pair", False):
                continue
            if wa.shape[0] != wb.shape[0] or wa.shape[1] != wb.shape[1] or wa.dtype != wb.dtype:
                continue
            w = torch.cat([wa, wb], dim=0)
            ma.weight_deq = w
            mb.weight_deq = w  # both point at the same fused weight
            ma.fused_pair = True
            mb.fused_pair = True
            ma.fused_offset = 0
            mb.fused_offset = wa.shape[0]
            ma.fused_total = w.shape[0]
            fused += 1
    return fused


def _repack_model_impl(model: nn.Module) -> int:
    """Original marlin repack loop (unchanged behavior)."""
    count = 0
    for name, module in model.named_modules():
        if not getattr(module, "nvfp4_enabled", False):
            continue
        if getattr(module, "nvfp4_mode", None) != "kernel":
            continue
        if getattr(module, "marlin_ready", False):
            continue
        w = getattr(module, "weight", None)
        if w is None or w.dtype != torch.uint8 or w.ndim != 2:
            continue
        if (w.shape[0], w.shape[1] * 2) in DEQ_SHAPES:
            continue  # Round4-B dispatch table: routed to cuBLAS dequant instead
        if w.shape[0] < _MARLIN_N_MIN:
            continue  # tiny-N layers keep Triton (marlin slower there)
        try:
            repack_linear(module)
            count += 1
        except Exception as e:
            print(f"[marlin] repack failed for {name}: {type(e).__name__}: {e}")
    print(f"[marlin] repacked {count} layers")
    return count


def marlin_linear(module: nn.Module, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor | None:
    """Run module forward via Marlin if applicable; None = not applicable."""
    if not getattr(module, "marlin_ready", False):
        return None
    # Marlin kernel needs REAL fp16 tensors (input_dtype=fp16 is not advisory —
    # bf16 input produces 1e-34 garbage). Convert in/out.
    orig_dtype = x.dtype
    if orig_dtype != torch.float16:
        x = x.to(torch.float16)
    if module.weight_triton is not None:
        # Triton copy retained (ADALLM_MARLIN_FREE_TRITON=0): use it for
        # prefill-scale batches where Triton is competitive
        M = x.numel() // x.shape[-1]
        if M > _MARLIN_M_MAX:
            return None
    out = apply_fp4_marlin_linear(
        x,
        module.weight,
        module.weight_scale,
        module.weight_marlin_global,
        module.workspace,
        module.marlin_n,
        module.marlin_k,
        bias=None,  # bias must be marlin-permuted to go inside the kernel;
        # cheaper to add outside (one vector add) than to permute
        input_dtype=torch.float16,
    )
    if out.dtype != orig_dtype:
        out = out.to(orig_dtype)
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out
