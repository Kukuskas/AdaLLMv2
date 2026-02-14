from __future__ import annotations

from functools import lru_cache
import torch


def _make_fp4_e2m1_lut() -> torch.Tensor:
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
    return torch.tensor(lut, dtype=torch.float32)


_FP4_E2M1_LUT = _make_fp4_e2m1_lut()


@lru_cache(8)
def _lut_for_device(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return _FP4_E2M1_LUT.to(device=device, dtype=dtype)


def get_nvfp4_group_size(quant_config: dict) -> int:
    try:
        return int(quant_config["quantization"]["group_size"])
    except Exception:
        return 16


def dequant_nvfp4(
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor | None,
    group_size: int = 16,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    if packed_weight.dtype != torch.uint8:
        raise ValueError("packed_weight must be uint8 for NVFP4 dequantization")
    if packed_weight.ndim != 2:
        raise ValueError("packed_weight must be 2D [out, in/2]")
    out_features, in_half = packed_weight.shape
    in_features = in_half * 2
    if in_features % group_size != 0:
        raise ValueError("in_features must be divisible by group_size")

    device = packed_weight.device
    lut = _lut_for_device(device, torch.float32)
    low = (packed_weight & 0x0F).to(torch.int64)
    high = (packed_weight >> 4).to(torch.int64)
    vals_low = lut[low]
    vals_high = lut[high]

    dequant = torch.empty((out_features, in_features), device=device, dtype=torch.float32)
    dequant[:, 0::2] = vals_low
    dequant[:, 1::2] = vals_high

    scale = weight_scale.float()
    if weight_scale_2 is not None:
        scale = scale * weight_scale_2.float()
    if scale.ndim == 1:
        scale = scale.view(out_features, 1)
    num_groups = in_features // group_size
    if scale.shape[1] not in (1, num_groups):
        raise ValueError("weight_scale shape does not match packed_weight")
    # Apply scales without materializing a full [out_features, in_features] scale tensor.
    dequant = dequant.view(out_features, num_groups, group_size)
    if scale.shape[1] == 1:
        dequant = dequant * scale.view(out_features, 1, 1)
    else:
        dequant = dequant * scale.view(out_features, num_groups, 1)
    return dequant.view(out_features, in_features).to(dtype)
