"""AdallmMarlinNvFp4LinearKernel: NvFp4LinearKernel registered on sm89.

Follows the base-class contract from
vllm/model_executor/kernels/linear/nvfp4/base.py (vLLM 0.32.0):

    is_supported(cls, compute_capability) -> (bool, reason|None)
    can_implement(cls, config) -> (bool, reason|None)
    process_weights_after_loading(layer) -> None
    apply_weights(layer, x, bias) -> Tensor

Differences from stock ``MarlinNvFp4LinearKernel``
(vllm/model_executor/kernels/linear/nvfp4/marlin.py):

  1. is_supported() gates to compute capability 89 (RTX 4090 / Ada).
  2. process_weights_after_loading() uses the VENDORED adallm repack
     (marlin_vendored.repack_linear): explicit scale_factor so
     smin*sf*128 >= 2 — stock ``prepare_fp4_layer_for_marlin`` computes
     ``_nvfp4_compute_scale_factor`` which returns 1.0 for fp16 activations
     and S0E5M3-zeroes adallm-scale-range (~2e-3) group scales.
  3. ModelOpt layers carry a fp32 ``weight_global_scale`` (amax/448 style):
     it is folded into the per-group scales before the repack so the whole
     scale path matches adallm's ``repack_linear`` convention.

This kernel wins the registry scan for W4A4 ModelOpt layers on sm89 (stock
FlashInfer TRT-LLM has no cc gate and would otherwise claim them). For
W4A16 the auto-selector force-picks stock Marlin before consulting the
registry — those checkpoints are handled by the AdallmModelOpt quant-config
steal (see scheme.py) instead.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.model_executor.kernels.linear.nvfp4.base import (
        NvFp4LinearLayerConfig,
    )

logger = logging.getLogger(__name__)


class AdallmMarlinNvFp4LinearKernel:
    """NVFP4 weight-only GEMM via adallm's Marlin path (sm89)."""

    def __init__(self, config: "NvFp4LinearLayerConfig") -> None:
        assert self.can_implement(config)[0]
        assert self.is_supported()[0]
        self.config = config

    def input_quant_key(self):
        return None  # weight-only (W4A16): no input quantization

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if compute_capability is None:
            try:
                from vllm.platforms import current_platform

                if not current_platform.is_cuda():
                    return False, "adallm kernels are CUDA-only"
                cc = current_platform.get_device_capability()
                compute_capability = cc.to_int() if cc is not None else None
            except Exception as e:
                return False, f"platform probe failed: {e}"
        if compute_capability != 89:
            return (
                False,
                f"adallm Marlin targets sm89 (Ada), got cc={compute_capability}",
            )
        from . import marlin_vendored

        if not marlin_vendored._IMPORT_OK:
            return False, "vendored marlin imports failed"
        return True, None

    @classmethod
    def can_implement(
        cls, config: "NvFp4LinearLayerConfig"
    ) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from . import marlin_vendored

        # ModelOpt layers: fold the fp32 weight_global_scale (amax/448
        # semantics) into the per-group scales so the effective scale the
        # kernel sees matches the checkpoint, then hand off to the adallm
        # repack (explicit sf; no scale_factor=1.0 forcing, no zeroing).
        if hasattr(layer, "weight_global_scale") and getattr(
            layer, "weight_scale", None
        ) is not None:
            gs = layer.weight_global_scale.detach()
            gs = float(gs.reshape(-1)[0].item()) if gs.numel() else 1.0
            layer.weight_scale = torch.nn.Parameter(
                (layer.weight_scale.detach().float() * gs).to(torch.float16),
                requires_grad=False,
            )
            delattr(layer, "weight_global_scale")
        if not hasattr(layer, "weight_scale_2"):
            layer.weight_scale_2 = torch.nn.Parameter(
                torch.ones(1, dtype=torch.float32, device=layer.weight.device),
                requires_grad=False,
            )
        layer.nvfp4_group_size = 16
        marlin_vendored.repack_linear(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from . import marlin_vendored

        return (
            self._fallback_matmul(layer, x, bias)
            if not getattr(layer, "marlin_ready", False)
            else marlin_vendored.marlin_linear(layer, x, bias)
        )

    @staticmethod
    def _fallback_matmul(layer, x, bias):
        """Graph-safe dequant fallback (profiling/dummy-weights passes only).

        Round11: the original version did `_E2M1_LUT.to(device)` and
        `.item()` on scale_2 — both create CPU<->CUDA syncs/copies that
        CRASH CUDA graph capture ("Cannot copy between CPU and CUDA tensors
        during CUDA graph capture"). Now: LUT cached on-device, scale_2
        consumed as a 0-dim GPU tensor, no host syncs.
        """
        import torch.nn.functional as F

        w8 = layer.weight
        N, K_half = w8.shape
        K = K_half * 2
        lo = (w8 & 0x0F).long()
        hi = (w8 >> 4).long()
        idx = torch.stack([lo, hi], dim=-1).reshape(N, K)

        lut = getattr(AdallmMarlinNvFp4LinearKernel, "_lut_cache", None)
        if lut is None or lut.device != x.device:
            from .tests_lut import _E2M1_LUT

            lut = _E2M1_LUT.to(x.device, non_blocking=False)
            AdallmMarlinNvFp4LinearKernel._lut_cache = lut
        base = lut[idx]

        s = layer.weight_scale
        if s.dtype == torch.float8_e4m3fn:
            s = s.float()
        s2 = getattr(layer, "weight_scale_2", None)
        if s2 is not None and s2.numel():
            s2v = s2.reshape(-1)[0].to(x.device, dtype=torch.float32)
        else:
            s2v = torch.ones((), device=x.device, dtype=torch.float32)
        w = (base.float() * s.float().repeat_interleave(16, dim=1) * s2v).to(
            x.dtype
        )
        out = F.linear(x, w)
        if bias is not None:
            out = out + bias
        return out


__all__ = ["AdallmMarlinNvFp4LinearKernel"]
