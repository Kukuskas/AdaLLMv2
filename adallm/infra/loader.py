import json
import os
import sys
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


_NORM_PLAIN_CONVENTION = False


def _is_aux_quant_tensor(name: str, keep_weight_scales: bool) -> bool:
    if name.endswith((".input_scale", ".k_scale", ".v_scale")):
        return True
    if name.endswith((".weight_scale", ".weight_scale_2")):
        return not keep_weight_scales
    # airagrp QAT per-group scales are consumed with their packed weight
    if name.endswith(".scales"):
        return True
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


def _quantize_nvfp4(weight: torch.Tensor, group_size: int = 16):
    """Quantize a float weight [N, K] to NVFP4 packed uint8 [N, K/2] with
    fp8_e4m3 per-group scales (e2m1 grid, same LUT as dequant_nvfp4)."""
    N, K = weight.shape
    w = weight.float().view(N, K // group_size, group_size)
    # e2m1 code values (unsigned magnitudes; sign bit separate)
    _E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=weight.device)
    sg = w.abs().amax(dim=-1, keepdim=True)
    sg = torch.where(sg == 0, torch.ones_like(sg), sg)
    # scale so max maps to 6.0, then quantize scale to fp8
    scale = (sg / 6.0)
    scale = scale.to(torch.float8_e4m3fn).float()
    scale = torch.where(scale == 0, torch.ones_like(scale) * 1e-6, scale)
    wn = w / scale
    # nearest e2m1 magnitude
    idx = torch.argmin((wn.abs().unsqueeze(-1) - _E2M1).abs(), dim=-1)
    mag = _E2M1[idx]
    sign = (w < 0).to(torch.uint8)
    # e2m1 nibble: bit3 = sign, bits0-2 = magnitude index
    code = torch.where(mag != 0, idx.to(torch.uint8), torch.zeros_like(sign))
    code = code | (sign << 3)
    # pack pairs
    flat = code.reshape(N, K // 2, 2)
    packed = (flat[:, :, 0] & 0x0F) | ((flat[:, :, 1] & 0x0F) << 4)
    return packed.contiguous(), scale.reshape(N, K // group_size).to(torch.float8_e4m3fn).half()


def _detect_plain_norm_convention(path: str) -> bool:
    """True if RMSNorm weights are plain multipliers (mean ~1) instead of
    Gemma-style offsets (mean ~0). Sniffs one input_layernorm weight."""
    import glob as _glob
    for file in _glob.glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for k in f.keys():
                if k.endswith("layers.0.input_layernorm.weight"):
                    t = f.get_tensor(k).float()
                    return bool(t.mean().item() > 0.5)
    return False


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    _NORM_PLAIN = _detect_plain_norm_convention(path)
    global _NORM_PLAIN_CONVENTION
    _NORM_PLAIN_CONVENTION = _NORM_PLAIN
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

    # ---- Qwen4Exp fast path ------------------------------------------------
    q4e_map = getattr(model, "q4e_map", None)
    if q4e_map is not None:
        return _load_qwen4_exp(model, path, q4e_map, show_progress)

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
                # airagrp QAT per-group scales: remap to ModelOpt's
                # weight_scale param name and convert fp8-bytes -> fp16.
                if weight_name.endswith(".scales"):
                    weight_name = weight_name[: -len(".scales")] + ".weight_scale"
                # airagrp QAT nesting: language_model.model.* vs the module
                # tree's model.language_model.* — remap the prefix.
                if weight_name.startswith("language_model.model."):
                    # language_model.model.layers.* -> model.language_model.layers.*
                    weight_name = "model.language_model." + weight_name[len("language_model.model."):]
                elif weight_name.startswith("language_model.lm_head"):
                    weight_name = weight_name[len("language_model."):]
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
                # scale tensors for modules without scale params (lm_head etc.
                # dequantized at load) must be dropped, not looked up. Scalars
                # (weight_scale_2) go through the plain scale_loader instead of
                # row-parallel narrow loaders.
                if weight_name.endswith((".weight_scale", ".weight_scale_2")):
                    try:
                        _sp2 = model.get_parameter(weight_name)
                        if _sp2.numel() == 1 and not hasattr(_sp2, "weight_loader"):
                            continue
                    except (KeyError, AttributeError):
                        continue
                # ModelOpt per-TENSOR fp8 scale (0-dim .weight_scale for FP8
                # modules): already applied during FP8 dequant/requant above.
                if weight_name.endswith(".weight_scale") and file_key.endswith(".weight_scale"):
                    _lw_probe = f.get_slice(file_key)
                    try:
                        _dims = len(_lw_probe.get_shape())
                    except Exception:
                        _dims = -1
                    if _dims == 0:
                        continue
                if _should_ignore_weight(weight_name):
                    continue
                loaded_weight = f.get_tensor(file_key)
                # airagrp QAT: RMSNorm weights stored in the PLAIN convention
                # (out = x_norm * w, weights centered at 1.0). This runtime's
                # GemmaRMSNorm uses out = x_norm * (1 + w). Convert w -> w-1.
                if _NORM_PLAIN_CONVENTION and weight_name.endswith(".weight"):
                    _n = weight_name
                    _gemma_site = (
                        _n.endswith("input_layernorm.weight")
                        or _n.endswith("post_attention_layernorm.weight")
                        or _n.endswith("q_norm.weight")
                        or _n.endswith("k_norm.weight")
                        # final model norm: "...language_model.norm.weight"
                        or (_n.endswith("language_model.norm.weight"))
                    )
                    if _gemma_site and loaded_weight.is_floating_point():
                        loaded_weight = (loaded_weight.float() - 1.0).to(loaded_weight.dtype)
                # General rule for kernel mode: ANY float weight destined for a
                # packed uint8 param (bf16 unquantized e.g. in_proj_a/b, FP8
                # mixed-precision groups) must be re-quantized to NVFP4.
                if (
                    loaded_weight.is_floating_point()
                    and weight_name.endswith(".weight")
                    and keep_weight_scales
                ):
                    try:
                        _p = model.get_parameter(weight_name)
                    except (KeyError, AttributeError):
                        _p = None
                    if _p is not None and _p.dtype == torch.uint8:
                        ws_name = weight_name + "_scale"
                        _ws = None
                        if loaded_weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                            if ws_name in all_keys:
                                _ws = f.get_tensor(ws_name).float()
                                if _ws.numel() == 1:
                                    loaded_weight = loaded_weight.to(target_dtype) * _ws.reshape(()).item()
                        _gs = get_nvfp4_group_size(quant_config) or 16
                        _packed, _sc = _quantize_nvfp4(loaded_weight.float(), group_size=_gs)
                        loaded_weight = _packed.to(_p.device)
                        try:
                            _sp = model.get_parameter(weight_name.replace(".weight", ".weight_scale"))
                            _sp.data.copy_(_sc.to(_sp.device))
                        except (KeyError, AttributeError):
                            pass
                if (
                    weight_name.endswith(".weight_scale")
                    and loaded_weight.dtype == torch.uint8
                ):
                    # airagrp QAT scales are raw fp8_e4m3 bytes
                    loaded_weight = loaded_weight.view(torch.float8_e4m3fn).to(torch.float16)
                # airagrp QAT conv1d layout: [out, kernel, 1] vs torch's
                # [out, 1, kernel] — transpose back.
                if (
                    weight_name.endswith("conv1d.weight")
                    and loaded_weight.ndim == 3
                    and loaded_weight.shape[1] != 1
                    and loaded_weight.shape[2] == 1
                ):
                    loaded_weight = loaded_weight.transpose(1, 2).contiguous()
                # FP8 (ModelOpt mixed-precision): weight is float8_e4m3 with a
                # scalar .weight_scale multiplier. Dequantize to target dtype.
                if loaded_weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    ws_name = weight_name + "_scale"
                    ws = f.get_tensor(ws_name).float() if ws_name in all_keys else None
                    loaded_weight = loaded_weight.to(target_dtype)
                    if ws is not None and ws.numel() == 1:
                        loaded_weight = loaded_weight * ws.reshape(()).item()
                    # kernel-mode packed params need NVFP4, not fp16: re-quantize
                    try:
                        _p = model.get_parameter(weight_name)
                    except (KeyError, AttributeError):
                        _p = None
                    if _p is not None and _p.dtype == torch.uint8 and keep_weight_scales:
                        _gs = get_nvfp4_group_size(quant_config) or 16
                        packed, sc = _quantize_nvfp4(loaded_weight.float(), group_size=_gs)
                        loaded_weight = packed.to(_p.device)
                        # stash scales for the module's weight_scale param
                        try:
                            _sp = model.get_parameter(weight_name.replace(".weight", ".weight_scale"))
                            _sp.data.copy_(sc.to(_sp.device))
                        except (KeyError, AttributeError):
                            pass
                # airagrp QAT naming: <module>.weight (U8 packed) + .scales
                # (equivalent to weight_scale). ModelOpt per-group scale may be
                # fp8_e4m3 with implicit *1.0 second-level scale.
                qat_scales_name = None
                if (
                    weight_name.endswith(".weight")
                    and (weight_name + "_scale") not in f.keys()
                ):
                    # try both remapped and original key namespaces
                    cand = weight_name[: -len(".weight")] + ".scales"
                    cand_orig = file_key[: -len(".weight")] + ".scales" if file_key.endswith(".weight") else None
                    hit = cand if cand in all_keys else (cand_orig if cand_orig and cand_orig in all_keys else None)
                    if hit:
                        qat_scales_name = hit
                        # airagrp QAT: weight stored as uint32 (8 nibbles/word)
                        if loaded_weight.dtype == torch.uint32:
                            loaded_weight = loaded_weight.view(torch.uint8).reshape(
                                loaded_weight.shape[0], -1
                            )
                use_nvfp4 = (
                    loaded_weight.dtype == torch.uint8
                    and weight_name.endswith(".weight")
                    and ((weight_name + "_scale") in f.keys() or qat_scales_name is not None)
                )
                _dequant_this = not keep_weight_scales
                if use_nvfp4 and keep_weight_scales and os.getenv("ADALLM_DEBUG_LOAD"):
                    print(f"[load] {weight_name}: packed-param-check", file=sys.stderr, flush=True)
                if use_nvfp4 and keep_weight_scales:
                    # kernel-mode global flag, but the DESTINATION param decides:
                    # non-packed params (lm_head/embed) force dequant of this tensor.
                    try:
                        _p = model.get_parameter(weight_name)
                        _packed_param = _p is not None and _p.dtype == torch.uint8
                    except (KeyError, AttributeError) as _e:
                        _packed_param = False
                        if os.getenv("ADALLM_DEBUG_LOAD"):
                            print(f"[load] {weight_name}: lookup FAILED {_type(_e).__name__}", file=sys.stderr, flush=True)
                    if os.getenv("ADALLM_DEBUG_LOAD"):
                        print(f"[load] {weight_name}: packed={_packed_param} dequant={not _packed_param}", file=sys.stderr, flush=True)
                    _dequant_this = not _packed_param
                if use_nvfp4 and _dequant_this:
                    if qat_scales_name is not None:
                        weight_scale = f.get_tensor(qat_scales_name)
                        # airagrp QAT scales: raw fp8_e4m3 bytes stored as uint8
                        if weight_scale.dtype == torch.uint8:
                            weight_scale = weight_scale.view(torch.float8_e4m3fn)
                    else:
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
                    if os.getenv("ADALLM_DEBUG_LOAD") and param.dim() >= 2 and loaded_weight.dim() >= 2 and param.shape != loaded_weight.shape:
                        print(f"[load-mismatch] {weight_name}: param {tuple(param.shape)} vs loaded {tuple(loaded_weight.shape)} dtypes {param.dtype}/{loaded_weight.dtype}", file=sys.stderr, flush=True)
                    weight_loader(param, loaded_weight)
                if show_progress:
                    pbar.update(1)
    if show_progress:
        pbar.close()
    # QAT-style checkpoints carry no per-tensor second-level scale; modules
    # created with torch.empty([]) would hold garbage -> NaN. Default to 1.0.
    import torch as _t
    with _t.no_grad():
        for module in model.modules():
            s2 = getattr(module, "weight_scale_2", None)
            if s2 is not None and s2.numel() == 1 and (_t.isnan(s2).any() or _t.isinf(s2).any() or s2.abs() > 1e12):
                s2.fill_(1.0)
    _init_nvfp4_bcol_scales(model)
    # Marlin repack (after all weights loaded; bcol init above is a no-op
    # unless NVFP4_FP8_BCOL is on, which is incompatible with Marlin)
    _marlin_on = os.getenv("ADALLM_MARLIN", "0").strip().lower() not in ("0", "false", "off", "")
    _tinygemm_on = os.getenv("ADALLM_TINYGEMM", "0").strip().lower() not in ("0", "false", "off", "")
    if _marlin_on or _tinygemm_on:
        from adallm.ops.marlin import repack_model
        repack_model(model)
    # lm_head NVFP4+Marlin repack (round2-C; off by default, env-gated)
    if os.getenv("ADALLM_LMHEAD_MARLIN", "0").strip().lower() not in ("0", "false", "off", ""):
        from adallm.ops.lmhead_marlin import repack_lm_head
        lm_head = getattr(model, "lm_head", None)
        if lm_head is None:
            print("[lmhead-marlin] no lm_head on model; skip", flush=True)
        elif getattr(model, "config", None) is not None and getattr(model.config, "tie_word_embeddings", False):
            # tied weights: quantizing lm_head would corrupt the embedding too
            print("[lmhead-marlin] tie_word_embeddings=True; skip (would corrupt embed)", flush=True)
        elif not repack_lm_head(lm_head):
            print("[lmhead-marlin] repack failed; keeping fp16 lm_head", flush=True)
        # the quantize+repack spike (transient fp32 chunks) poisons the CUDA
        # peak-memory high-water mark used for KV pool sizing → reset it
        torch.cuda.reset_peak_memory_stats()


def _load_qwen4_exp(model: nn.Module, path: str, q4e_map, show_progress: bool):
    """Dedicated loader for the Qwen4Exp checkpoint layout.

    - 297,984 expert tensors (512 experts x 48 layers x 3 projs x 4 tensors):
      routed by regex to the EP params; REMOTE experts are skipped without
      materializing the tensor (checkpoint file shard is still read by
      safetensors' lazy get_tensor, but only local rows are copied).
    - 128 PLE shard tensors (~51GB): skipped entirely (mmap at runtime).
    - everything else: dense BF16 / packed-NVFP4 copies by exact name.
    """
    index_path = os.path.join(path, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    # group names by file for single-pass reads
    by_file: dict[str, list[str]] = {}
    for name, fname in weight_map.items():
        by_file.setdefault(fname, []).append(name)

    n_local = 0
    n_skip = 0
    pbar = tqdm(total=len(weight_map), desc="Loading qwen4_exp", disable=not show_progress, dynamic_ncols=True)
    for fname in sorted(by_file.keys()):
        fpath = os.path.join(path, fname)
        with safe_open(fpath, framework="pt", device="cpu") as f:
            for weight_name in sorted(by_file[fname]):
                mapped = q4e_map(weight_name)
                if mapped is not None:
                    kind = mapped[0]
                    if kind == "ple_skip":
                        n_skip += 1
                        pbar.update(1)
                        continue
                    if kind == "expert":
                        _, layer_idx, gid, new_name = mapped
                        if not model.expert_is_local(layer_idx, gid):
                            n_skip += 1
                            pbar.update(1)
                            continue
                        t = f.get_tensor(weight_name)
                        if t.dtype == torch.uint8 or weight_name.endswith("weight_scale_2"):
                            pass  # packed weight / fp32 scale_2: direct
                        elif weight_name.endswith(".weight_scale"):
                            t = t.view(torch.float8_e4m3fn).to(torch.float16)
                        model.load_expert_tensor(new_name, gid, t)
                        n_local += 1
                        pbar.update(1)
                        continue
                    if kind == "ple_scale":
                        model.load_ple_scale(mapped[1], f.get_tensor(weight_name))
                        pbar.update(1)
                        continue
                # ---- dense path (everything else) ----
                if _should_ignore_weight(weight_name):
                    n_skip += 1
                    pbar.update(1)
                    continue
                if weight_name.startswith("mtp."):
                    n_skip += 1
                    pbar.update(1)
                    continue
                loaded_weight = f.get_tensor(weight_name)
                try:
                    param = model.get_parameter(weight_name)
                except (KeyError, AttributeError):
                    param = model.get_buffer(weight_name)
                if param is None:
                    raise KeyError(f"q4e dense param not found: {weight_name}")
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                n_local += 1
                pbar.update(1)
    pbar.close()
    if show_progress:
        print(f"[q4e-load] loaded={n_local} skipped={n_skip}", flush=True)
    # attach the mmap'd PLE table file
    ple_path = os.path.join(path, "model-fp8-mtp-ple.safetensors")
    if os.path.isfile(ple_path):
        model.attach_ple_file(ple_path)
    else:
        print("[q4e-load] WARNING: model-fp8-mtp-ple.safetensors missing; PLE table unavailable", flush=True)
    return None


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
