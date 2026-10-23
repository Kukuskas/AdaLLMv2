"""AdallmMarlinScheme: vLLM scheme stub wrapping adallm's Marlin NVFP4 GEMM.

Status: SKELETON. Signatures mirror the closest live implementations in
vLLM 0.32.0:

  - create_weights/process_weights_after_loading/apply_weights follow
    ``CompressedTensorsW4A4Fp4``
    (vllm/model_executor/layers/quantization/compressed_tensors/schemes/
     compressed_tensors_w4a4_nvfp4.py)
  - the Marlin path itself follows ``MarlinNvFp4LinearKernel``
    (vllm/model_executor/kernels/linear/nvfp4/marlin.py) which delegates to
    ``prepare_fp4_layer_for_marlin`` + ``apply_fp4_marlin_linear``
    (vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py)

The numerics that differentiate adallm live in the vendored
:mod:`.marlin_vendored` (``repack_linear``: explicit scale_factor so
smin*sf*128 >= 2 with global scale divided by sf; weight_scale_2 folded
into per-group scales). See docs/PORT_PLAN.md for the port order.

Runtime selection seam (preferred over a full scheme swap): the plugin
registers :class:`AdallmMarlinNvFp4LinearKernel` into vLLM's
``register_linear_kernel(..., kernel_type="nvfp4")`` registry, where
``is_supported`` returns True on sm89 (Ada) and the kernel reuses the
vendored repack. This routes ``modelopt_fp4`` W4A16 layers to adallm Marlin
WITHOUT replacing the quantization config, checkpoint loader, or MoE path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.nn.parameter import Parameter

if TYPE_CHECKING:
    from collections.abc import Callable

    from vllm.model_executor.kernels.linear.nvfp4.base import (
        NvFp4LinearKernel,
        NvFp4LinearLayerConfig,
    )


class AdallmMarlinScheme:
    """W4A16 NVFP4 Marlin scheme (adallm scale-path fixes) — STUB.

    Implements the CompressedTensorsScheme-style interface that vLLM's
    linear methods call. All bodies raise NotImplementedError except the
    weight layout (which matches the NVFP4 modelopt layout so the stock
    weight_loader works unchanged).
    """

    group_size = 16

    def __init__(self, *, use_a16: bool = True) -> None:
        # a16 = weight-only (bf16/fp16 activations) — the adallm regime.
        self.use_a16 = use_a16

    @classmethod
    def get_min_capability(cls) -> int:
        # Marlin FP4 runs on sm75+; adallm targets sm89 (RTX 4090).
        return 89

    # ------------------------------------------------------------------
    # create_weights — signature copied from CompressedTensorsW4A4Fp4.
    # Registers the modelopt NVFP4 layout so the stock vLLM weight loaders
    # fill it: uint8 packed weight [N, K/2], fp8-e4m3 group scales
    # [N, K/16], fp32 global scales.
    # ------------------------------------------------------------------
    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: "Callable",
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError("TODO(port-plan phase 1)")

    # ------------------------------------------------------------------
    # process_weights_after_loading — repack to Marlin layout using the
    # vendored repack_linear() (explicit scale_factor + weight_scale_2
    # folding), then stash the Marlin workspace on the layer.
    # ------------------------------------------------------------------
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        raise NotImplementedError("TODO(port-plan phase 1)")

    # ------------------------------------------------------------------
    # apply_weights — call apply_fp4_marlin_linear with input_dtype=fp16
    # (bf16 input produces garbage; convert in/out as marlin_linear does).
    # ------------------------------------------------------------------
    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("TODO(port-plan phase 1)")


def adallm_marlin_kernel_available() -> bool:
    """True when the vendored marlin module imported against a compatible
    vLLM (0.32 kernel layout) and Marlin FP4 is supported on this platform.
    CPU-safe: returns False instead of raising when vLLM is absent."""
    from . import marlin_vendored

    if not marlin_vendored._IMPORT_OK:
        return False
    try:
        return bool(marlin_vendored.is_fp4_marlin_supported())
    except Exception:
        return False


__all__ = [
    "AdallmMarlinScheme",
    "adallm_marlin_kernel_available",
]
