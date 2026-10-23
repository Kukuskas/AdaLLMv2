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

        return marlin_vendored.marlin_linear(layer, x, bias)


__all__ = ["AdallmMarlinNvFp4LinearKernel"]
