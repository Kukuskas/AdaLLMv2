import json
import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
from tqdm import tqdm

from adallm.infra.nvfp4 import dequant_nvfp4, get_nvfp4_group_size
from adallm.infra.quant import get_quant_config


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _load_hf_quant_config(path: str) -> dict:
    quant_path = os.path.join(path, "hf_quant_config.json")
    if not os.path.isfile(quant_path):
        return {}
    with open(quant_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_aux_quant_tensor(name: str, keep_weight_scales: bool) -> bool:
    if name.endswith((".input_scale", ".k_scale", ".v_scale")):
        return True
    if name.endswith((".weight_scale", ".weight_scale_2")):
        return not keep_weight_scales
    return False


def _should_ignore_weight(name: str) -> bool:
    if name.startswith(("vision_tower.", "multi_modal_projector.", "model.visual.")):
        return True
    if ".vision_tower." in name or ".multi_modal_projector." in name or ".visual." in name:
        return True
    if ".mtp." in name or name.startswith("mtp."):
        return True
    return False


def _estimate_total_weights(path: str, keep_weight_scales: bool) -> int | None:
    index_path = os.path.join(path, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return None
    with open(index_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f).get("weight_map", {})
    total = 0
    for name in weight_map.keys():
        if _is_aux_quant_tensor(name, keep_weight_scales):
            continue
        if _should_ignore_weight(name):
            continue
        total += 1
    return total


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    quant_config = _load_hf_quant_config(path)
    group_size = get_nvfp4_group_size(quant_config)
    qcfg = get_quant_config()
    keep_weight_scales = qcfg.enabled and qcfg.mode == "kernel"
    target_dtype = torch.get_default_dtype()
    show_progress = os.getenv("ADALLM_DISABLE_PROGRESS", "").strip().lower() not in ("1", "true", "yes", "on")
    try:
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            show_progress = False
    except Exception:
        pass
    total = _estimate_total_weights(path, keep_weight_scales) if show_progress else None
    pbar = tqdm(
        total=total,
        desc="Loading weights",
        disable=not show_progress,
        dynamic_ncols=True,
    )
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            all_keys = set(f.keys())
            for weight_name in f.keys():
                # key in the actual file (differs from weight_name after remap)
                file_key = weight_name
                # QUASAR W4A4 static activation scales: unused by this runtime
                # (FP8 activation quant is dynamic). Skip entirely.
                if weight_name.endswith(".input_global_scale"):
                    continue
                # Per-tensor global scale: merged into weight_scale_2 by the
                # .weight_packed branch below; skip the standalone key.
                if weight_name.endswith(".weight_global_scale"):
                    continue
                # QUASAR-QAT naming: <module>.weight_packed (+ _scale/_global_scale).
                # Remap to the ModelOpt convention the rest of the loader expects:
                #   weight_packed -> weight (uint8 packed)
                #   weight_scale (fp8 per-group) stays, dequantized to fp16 below
                #   weight_global_scale -> weight_scale_2 (fp32 scalar)
                orig_name = None
                if weight_name.endswith(".weight_packed"):
                    orig_name = weight_name
                    weight_name = weight_name[: -len(".weight_packed")] + ".weight"
                    # global scale key is derived from the REMAPPED name:
                    # <base>.weight_global_scale (not ..._packed_global_scale)
                    global_scale_name = weight_name + "_global_scale"
                    if global_scale_name in all_keys:
                        gs = f.get_tensor(global_scale_name).float().reshape(()).item()
                        if gs == 0:
                            gs = 1.0
                        # QUASAR global scale semantics: effective scale =
                        # weight_scale_fp8 / global_scale. ModelOpt's scale_2 is
                        # a MULTIPLIER, so store the reciprocal.
                        try:
                            param = model.get_parameter(weight_name + "_scale_2")
                            param.data.copy_(torch.tensor(1.0 / gs))
                        except (KeyError, AttributeError):
                            pass  # dequant mode: no scale_2 param; merged below
                if _is_aux_quant_tensor(weight_name, keep_weight_scales):
                    # These are handled alongside the packed weight or ignored.
                    continue
                if _should_ignore_weight(weight_name):
                    continue
                loaded_weight = f.get_tensor(file_key)
                use_nvfp4 = (
                    loaded_weight.dtype == torch.uint8
                    and weight_name.endswith(".weight")
                    and (weight_name + "_scale") in f.keys()
                )
                if use_nvfp4 and not keep_weight_scales:
                    weight_scale = f.get_tensor(weight_name + "_scale")
                    weight_scale_2_name = weight_name + "_scale_2"
                    weight_scale_2 = f.get_tensor(weight_scale_2_name) if weight_scale_2_name in f.keys() else None
                    if weight_scale_2 is None:
                        # QUASAR: per-tensor scale lives in <name>.weight_global_scale
                        # (derived from the remapped name), with DIVISION semantics.
                        gs_name = weight_name + "_global_scale"
                        if gs_name in f.keys():
                            gs = f.get_tensor(gs_name).float().reshape(()).item()
                            weight_scale_2 = torch.tensor(1.0 / (gs if gs != 0 else 1.0))
                    loaded_weight = dequant_nvfp4(
                        loaded_weight,
                        weight_scale,
                        weight_scale_2,
                        group_size=group_size,
                        dtype=target_dtype,
                    )
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        try:
                            param = model.get_parameter(param_name)
                        except (KeyError, AttributeError):
                            param = model.get_buffer(param_name)
                        if param is None:
                            raise KeyError(f"Parameter or buffer not found: {param_name}")
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight, shard_id)
                        break
                else:
                    try:
                        param = model.get_parameter(weight_name)
                    except (KeyError, AttributeError):
                        param = model.get_buffer(weight_name)
                    if param is None:
                        raise KeyError(f"Parameter or buffer not found: {weight_name}")
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                if show_progress:
                    pbar.update(1)
    if show_progress:
        pbar.close()
    _init_nvfp4_bcol_scales(model)


def _init_nvfp4_bcol_scales(model: nn.Module):
    if os.getenv("NVFP4_FP8_BCOL", "0").strip().lower() in ("0", "false", "off"):
        return
    # Use a safe upper bound for LUT magnitude to avoid saturation.
    lut_abs_max = 6.0
    for module in model.modules():
        if not getattr(module, "nvfp4_enabled", False):
            continue
        if getattr(module, "nvfp4_mode", None) != "kernel":
            continue
        if not hasattr(module, "weight_scale") or not hasattr(module, "weight_scale_2"):
            continue
        weight_scale = module.weight_scale
        if weight_scale is None or weight_scale.numel() == 0:
            continue
        with torch.no_grad():
            scale_max = weight_scale.max(dim=1).values
            scale2 = module.weight_scale_2
            if scale2 is None or scale2.numel() == 0:
                scale2 = torch.tensor(1.0, device=scale_max.device, dtype=scale_max.dtype)
            else:
                scale2 = scale2.to(scale_max.dtype)
            if scale2.numel() == 1:
                bcol = scale_max * scale2
            elif scale2.numel() == scale_max.numel():
                bcol = scale_max * scale2
            else:
                raise ValueError("weight_scale_2 has unsupported shape for bcol scaling")
            bcol = bcol * lut_abs_max
            bcol = torch.where(bcol == 0, torch.ones_like(bcol), bcol)
            module.weight_scale_col = bcol
