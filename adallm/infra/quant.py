from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class QuantConfig:
    enabled: bool = False
    algo: str | None = None
    group_size: int = 16
    mode: str = "dequant"  # "dequant" or "kernel"
    kv_cache_dtype: torch.dtype = torch.float16
    hybrid_kv_cache: bool = False
    kv_cache_sink_blocks: int = 0
    kv_cache_recent_blocks: int = 0
    fp8_decode_kernel_enabled: bool = True
    fp8_decode_kernel_min_ctx: int = 0


_QUANT_CONFIG = QuantConfig()


def get_quant_config() -> QuantConfig:
    return _QUANT_CONFIG


def _parse_kv_cache_dtype(value: str) -> torch.dtype:
    v = value.lower()
    if v in ("fp16", "float16", "f16"):
        return torch.float16
    if v in ("bf16", "bfloat16"):
        return torch.bfloat16
    if v in ("fp8", "float8", "e4m3"):
        return torch.float8_e4m3fn
    return torch.float16


def set_quant_config(hf_config, config) -> None:
    qc = getattr(hf_config, "quantization_config", None)
    algo = None
    group_size = None
    kv_bits = None
    if isinstance(qc, dict):
        algo = qc.get("quant_algo")
        if algo is None and "quantization" in qc:
            algo = qc["quantization"].get("quant_algo")
        try:
            group_size = qc["config_groups"]["group_0"]["weights"]["group_size"]
        except Exception:
            group_size = None
        try:
            kv_bits = qc["kv_cache_scheme"]["num_bits"]
        except Exception:
            kv_bits = None

        # QUASAR-QAT checkpoints: compressed-tensors nvfp4-pack-quantized (W4A4),
        # no quant_algo field. Detect via format + group_size 16.
        if algo is None and qc.get("format") == "nvfp4-pack-quantized":
            algo = "NVFP4"

    _QUANT_CONFIG.enabled = (algo == "NVFP4")
    _QUANT_CONFIG.algo = algo
    if group_size is not None:
        _QUANT_CONFIG.group_size = int(group_size)
    _QUANT_CONFIG.mode = getattr(config, "nvfp4_mode", "dequant")
    kv_dtype = getattr(config, "kv_cache_dtype", "fp16")
    if kv_dtype == "fp16" and kv_bits == 8:
        kv_dtype = "fp8"
    _QUANT_CONFIG.kv_cache_dtype = _parse_kv_cache_dtype(kv_dtype)
    _QUANT_CONFIG.hybrid_kv_cache = bool(getattr(config, "hybrid_kv_cache", False))
    _QUANT_CONFIG.kv_cache_sink_blocks = int(getattr(config, "kv_cache_sink_blocks", 0))
    _QUANT_CONFIG.kv_cache_recent_blocks = int(getattr(config, "kv_cache_recent_blocks", 0))
    _QUANT_CONFIG.fp8_decode_kernel_enabled = bool(getattr(config, "fp8_decode_kernel_enabled", True))
    _QUANT_CONFIG.fp8_decode_kernel_min_ctx = int(getattr(config, "fp8_decode_kernel_min_ctx", 2048))
