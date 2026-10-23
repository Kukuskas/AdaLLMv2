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
     On sm89 the stock chain resolves to slow backends for W4A4
     (FlashInfer TRT-LLM dequant or emulation); stock Marlin is our
     fallback elsewhere, so we only claim Ada.
  2. process_weights_after_loading() will use the VENDORED repack
     (explicit scale_factor; weight_scale_2 folding) instead of
     prepare_fp4_layer_for_marlin — implemented in port-plan phase 1;
     currently raises NotImplementedError.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.model_executor.kernels.linear.nvfp4.base import (
        NvFp4LinearLayerConfig,
    )


class AdallmMarlinNvFp4LinearKernel:
    """STUB — registration target; GEMM body lands in port-plan phase 1."""

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
        raise NotImplementedError("TODO(port-plan phase 1): vendored repack")

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("TODO(port-plan phase 1): apply_fp4_marlin_linear")


__all__ = ["AdallmMarlinNvFp4LinearKernel"]
