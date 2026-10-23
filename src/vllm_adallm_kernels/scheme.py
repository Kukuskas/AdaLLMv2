"""Quantization configs: adallm NVFP4 as vLLM custom quant methods.

Two registered methods (both checked BEFORE stock overrides in
``ModelConfig._verify_quantization`` — custom methods are placed at the
front of the scan list, vLLM 0.32 ``config/model.py`` ~L1410):

``adallm_nvfp4``
    adallm QAT checkpoints — ``config.json: quantization_config =
    {mode: "nvfp4", bits: 4, group_size: 16}``. Weights stored as
    uint32-packed fp4 + uint8-packed fp8-e4m3 group scales (``.scales``);
    feed the checkpoint through ``tools/convert_adallm_qat.py`` first
    (uint8 weight, fp16 ``weight_scale``, unfused vLLM projection names).

``adallm_modelopt_fp4``
    Detection steal for STOCK ModelOpt NVFP4 checkpoints
    (hf_quant_config.json with quant_algo containing FP4): routes them to
    our LinearMethod so the adallm Marlin scale path (explicit
    scale_factor — stock ``prepare_fp4_layer_for_marlin`` zeroes
    adallm-scale-range group scales) runs instead of the stock one.

The linear method registers the SAME parameter layout as stock
``KNvfp4Static`` (uint8 packed weight, fp8-e4m3 group scales, fp32
weight_scale_2) so the stock v2 weight loaders, TP slicing and the
Qwen3.5 fused-projection mapper all work unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.nn.parameter import Parameter

logger = logging.getLogger(__name__)

try:
    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.parameter import (
        ModelWeightParameter,
        PerTensorScaleParameter,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        is_layer_skipped,
    )

    _VLLM_OK = True
except Exception as _e:  # pragma: no cover
    _VLLM_OK = False
    _IMPORT_ERR = _e


# ---------------------------------------------------------------------------
# Linear method (shared by both configs)
# ---------------------------------------------------------------------------

class _BroadcastScaleParameter(PerTensorScaleParameter):
    """Per-tensor scale that accepts a 0-dim scalar for any partition count
    (fused checkpoints ship ONE weight_scale_2 for a 5-way-fused in_proj)
    and broadcasts it to every logical partition."""

    def _assert_and_load(self, loaded_weight: torch.Tensor):
        if loaded_weight.ndim == 0 and self.data.ndim >= 1:
            self.data.fill_(loaded_weight)
            return
        if self.data.shape != loaded_weight.shape and self.data.numel() == 1:
            self.data.copy_(loaded_weight.reshape(()))
            return
        super()._assert_and_load(loaded_weight)


class AdallmMarlinLinearMethod:
    """W4A16 NVFP4 linear method: stock layout params + adallm repack.

    Params mirror KNvfp4Static.create_weights (modelopt.py): uint8 packed
    weight [N, K/2] (input_dim=1, output_dim=0), fp8-e4m3 [N, K/16] group
    scales, fp32 weight_scale_2 per logical partition. The Marlin repack
    (vendored, adallm numerics: explicit sf, no scale_factor=1.0 forcing)
    runs in process_weights_after_loading.
    """

    group_size = 16

    def __init__(self, quant_config: "AdallmMarlinConfigBase"):
        self.quant_config = quant_config
        self._kernel = None

    # -- kernel (lazy: no CUDA at plugin-registration time) -----------------
    @property
    def kernel(self):
        if self._kernel is None:
            from .kernel import AdallmMarlinNvFp4LinearKernel
            from vllm.model_executor.kernels.linear.nvfp4.base import (
                NvFp4LinearLayerConfig,
            )

            ok, reason = AdallmMarlinNvFp4LinearKernel.is_supported()
            if not ok:
                raise RuntimeError(f"adallm marlin kernel unsupported: {reason}")
            self._kernel = AdallmMarlinNvFp4LinearKernel(
                config=NvFp4LinearLayerConfig()
            )
        return self._kernel

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = sum(output_partition_sizes)
        layer.params_dtype = params_dtype

        N = layer.output_size_per_partition
        K = input_size_per_partition
        g = self.group_size
        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty((N, K // 2), dtype=torch.uint8),
                weight_loader=weight_loader,
                input_dim=1,
                output_dim=0,
            ),
        )
        layer.register_parameter(
            "weight_scale",
            ModelWeightParameter(
                data=torch.full((N, K // g), float("nan"), dtype=torch.float8_e4m3fn),
                weight_loader=weight_loader,
                input_dim=1,
                output_dim=0,
            ),
        )
        layer.register_parameter(
            "weight_scale_2",
            _BroadcastScaleParameter(
                data=torch.ones(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            ),
        )
        # W4A4-format checkpoints carry a per-tensor input_scale; our path is
        # W4A16 (bf16/fp16 activations) so it is loaded and ignored.
        layer.register_parameter(
            "input_scale",
            _BroadcastScaleParameter(
                data=torch.ones(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            ),
        )
        # instantiate the kernel now that we're on the hot path (CUDA probe)
        _ = self.kernel

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if torch.isnan(layer.weight_scale.float()).any():
            raise RuntimeError(
                "adallm_nvfp4: weight_scale never loaded (NaN sentinel)"
            )
        if torch.isnan(layer.weight_scale_2.float()).any():
            raise RuntimeError(
                "adallm_nvfp4: weight_scale_2 never loaded (NaN sentinel)"
            )
        # KNvfp4Static semantics: fold per-partition scale_2 into a single
        # global multiplier on the group scales (they must all agree).
        s2 = float(layer.weight_scale_2.detach().float().max().item())
        # fp8e4m3 group scales -> fp16, global folded in (adallm convention:
        # dequant = e2m1 * group_scale, weight_scale_2 becomes 1.0)
        layer.weight_scale = Parameter(
            (layer.weight_scale.detach().float() * s2).to(torch.float16),
            requires_grad=False,
        )
        layer.weight_scale_2 = Parameter(
            torch.ones(1, dtype=torch.float32, device=layer.weight.device),
            requires_grad=False,
        )
        self.kernel.process_weights_after_loading(layer)

    def apply(self, layer, x, bias=None):
        return self.kernel.apply_weights(layer, x, bias)


# Route loading through the v2 path (BasevLLMParameter semantics; the v1
# MergedColumnParallelLinear loader cannot broadcast 0-dim scale_2 scalars
# into multi-partition fused layers). Registered AFTER the class definition.
try:
    from vllm.model_executor.layers.linear import (
        register_weight_loader_v2_supported_method,
    )

    register_weight_loader_v2_supported_method(AdallmMarlinLinearMethod)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

class AdallmMarlinConfigBase(QuantizationConfig):
    """Shared base."""

    def __init__(self, group_size: int = 16, exclude_modules=None):
        super().__init__()
        self.group_size = group_size
        self.exclude_modules: list[str] = list(exclude_modules or ["lm_head"])

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["config.json"]

    def is_layer_skipped(self, prefix: str) -> bool:
        return is_layer_skipped(prefix, self.exclude_modules, self.packed_modules_mapping)

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            if self.is_layer_skipped(prefix):
                from vllm.model_executor.layers.linear import (
                    UnquantizedLinearMethod,
                )

                return UnquantizedLinearMethod()
            return AdallmMarlinLinearMethod(self)
        return None


def _register() -> None:
    if not _VLLM_OK:
        return

    @register_quantization_config("adallm_nvfp4")
    class AdallmNvFp4Config(AdallmMarlinConfigBase):
        """adallm QAT NVFP4 (pre-converted layout: uint8 + fp16 scales on disk)."""

        def get_name(self):
            return "adallm_nvfp4"

        @classmethod
        def from_config(cls, config: dict[str, Any]) -> "AdallmNvFp4Config":
            return cls(group_size=int(config.get("group_size", 16)))

        @classmethod
        def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
            cfg = hf_quant_cfg or {}
            if (
                cfg.get("mode") == "nvfp4"
                and int(cfg.get("bits", 4)) == 4
                and (user_quant is None or user_quant == "adallm_nvfp4")
            ):
                return "adallm_nvfp4"
            return None

    @register_quantization_config("adallm_modelopt_fp4")
    class AdallmModelOptNvFp4Config(AdallmMarlinConfigBase):
        """Detection steal: stock ModelOpt NVFP4 checkpoints -> our kernel."""

        def get_name(self):
            return "adallm_modelopt_fp4"

        @classmethod
        def from_config(cls, config: dict[str, Any]) -> "AdallmModelOptNvFp4Config":
            q = config.get("quantization", config) if isinstance(config, dict) else {}
            exclude = q.get("exclude_modules") if isinstance(q, dict) else None
            group = q.get("group_size", 16) if isinstance(q, dict) else 16
            if exclude:
                # checkpoint excludes use the HF namespace (e.g. Nemotron's
                # "backbone.layers.N..."); get_quant_method sees vLLM prefixes
                exclude = [str(e).replace("backbone.", "model.") for e in exclude]
                exclude.append("lm_head")
            else:
                exclude = ["lm_head"]
            return cls(group_size=int(group), exclude_modules=exclude)

        @classmethod
        def get_config_filenames(cls) -> list[str]:
            return ["hf_quant_config.json", "config.json"]

        @classmethod
        def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
            algo = _extract_fp4_algo(hf_quant_cfg)
            if algo is not None and ("NVFP4" in algo or "FP4" in algo):
                if user_quant is None or user_quant in (
                    "adallm_modelopt_fp4",
                    "modelopt_fp4",
                ):
                    return "adallm_modelopt_fp4"
            return None

    globals()["AdallmNvFp4Config"] = AdallmNvFp4Config
    globals()["AdallmModelOptNvFp4Config"] = AdallmModelOptNvFp4Config


def _extract_fp4_algo(hf_quant_cfg) -> str | None:
    """ModelOptNvFp4Config._extract_modelopt_quant_algo equivalent (light).

    hf_quant_config.json layout: {"producer": {...}, "quantization":
    {"quant_algo": "NVFP4", ...}}; some checkpoints also put quant_algo at
    the top level or in config.json's quantization_config.
    """
    if hf_quant_cfg is None:
        return None
    for d in (
        hf_quant_cfg,
        hf_quant_cfg.get("quantization") or {},
    ):
        if isinstance(d, dict) and d.get("quant_algo"):
            return str(d["quant_algo"])
    return None


_register()


__all__ = [
    "AdallmMarlinLinearMethod",
    "AdallmNvFp4Config",
    "AdallmModelOptNvFp4Config",
    "adallm_marlin_kernel_available",
]


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
