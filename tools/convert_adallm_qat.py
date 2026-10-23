"""Convert an adallm QAT NVFP4 checkpoint to the vLLM-native layout served
by the adallm_nvfp4 quant method of vllm-adallm-kernels.

Input  (adallm QAT, e.g. Qwen3.8-27B-QAT-NVFP4):
    language_model.model.layers.N.<proj>.weight   uint32 [N, K/8] (8 nibbles)
    language_model.model.layers.N.<proj>.scales   uint8  [N, K/16] fp8e4m3 bytes
    norms in PLAIN convention (out = x_norm * w, mean ~1)
    language_model.lm_head.weight                 fp4-packed (dequantized here)
    vision_tower.* / language_model.mtp.*         dropped (text-only serve)

Output (vLLM Qwen3_5ForCausalLM layout, TP=1):
    model.layers.N.<unfused proj>.weight          uint8 [N, K/2]
    model.layers.N.<proj>.weight_scale            fp8e4m3 [N, K/16]
    model.layers.N.<proj>.weight_scale_2          fp32 [1] = 1.0
    model.embed_tokens.weight                     bf16
    lm_head.weight                                bf16 (dequantized)
    norms shifted to Gemma convention (w - 1)
    config.json: architectures=[Qwen3_5ForCausalLM], model_type=qwen3_5_text,
                 quantization_config={mode:nvfp4,...} kept for detection

Usage:
    python tools/convert_adallm_qat.py <src> <dst>
"""

from __future__ import annotations

import json
import os
import shutil
import sys

import torch
from safetensors.torch import safe_open, save_file

_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

NORM_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "q_norm.weight",
    "k_norm.weight",
)


def dequant(w_u32: torch.Tensor, scales_u8: torch.Tensor, group: int = 16) -> torch.Tensor:
    """uint32-packed [N, K/8] + uint8 fp8e4m3 scales [N, K/16] -> fp32 [N, K]."""
    w8 = w_u32.view(torch.uint8).reshape(w_u32.shape[0], -1)  # [N, K/2]
    N, K_half = w8.shape
    K = K_half * 2
    lo = (w8 & 0x0F).long()
    hi = (w8 >> 4).long()
    idx = torch.stack([lo, hi], dim=-1).reshape(N, K)
    base = _E2M1_LUT[idx]
    s = scales_u8.view(torch.float8_e4m3fn).float()
    s2 = s.repeat_interleave(group, dim=1)
    return base * s2


def main(src: str, dst: str) -> None:
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)

    # ---- config: text-only Qwen3_5ForCausalLM -----------------------------
    text = dict(cfg["text_config"])
    text["architectures"] = ["Qwen3_5ForCausalLM"]
    text["model_type"] = "qwen3_5_text"
    text["quantization_config"] = cfg["quantization_config"]
    text["torch_dtype"] = "bfloat16"
    for k in ("preprocessor_config.json", "video_preprocessor_config.json",
              "chat_template.jinja"):
        pass  # copied below
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(text, f, indent=2)

    # tokenizer + chat template
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                 "merges.txt", "generation_config.json", "chat_template.jinja",
                 "preprocessor_config.json"):
        p = os.path.join(src, name)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(dst, name))

    # ---- weights -----------------------------------------------------------
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    files = sorted(set(wm.values()))

    out: dict[str, torch.Tensor] = {}
    out_files: list[str] = []

    def flush(name: str):
        if out:
            fn = f"model-{len(out_files)+1:05d}-of-00008.safetensors"
            save_file(out, os.path.join(dst, fn))
            out_files.append(fn)
            out.clear()

    def add(k: str, t: torch.Tensor):
        # shard at ~4 GB
        nbytes = sum(t.numel() * t.element_size() for t in out.values()) + (
            t.numel() * t.element_size()
        )
        if nbytes > 4 * 1024**3:
            flush("size")
        out[k] = t.contiguous()

    for fn in files:
        with safe_open(os.path.join(src, fn), "pt") as f:
            keys = list(f.keys())
            for key in keys:
                if key.startswith(("vision_tower.", "language_model.mtp.")):
                    continue
                t = f.get_tensor(key)
                # prefix remap
                if key.startswith("language_model.model."):
                    name = "model." + key[len("language_model.model."):]
                elif key.startswith("language_model."):
                    name = key[len("language_model."):]
                else:
                    name = key

                if name.endswith(".scales"):
                    continue  # handled with its weight
                if name.endswith("norm.weight") and any(
                    name.endswith(s) for s in NORM_SUFFIXES
                ):
                    add(name, (t.float() - 1.0).to(t.dtype))  # Gemma shift
                elif name.endswith("conv1d.weight") and t.ndim == 3 and t.shape[1] != 1:
                    add(name, t.transpose(1, 2).contiguous())
                elif name in ("model.embed_tokens.weight",) or name == "lm_head.weight":
                    # dequant packed embed/lm_head
                    wkey = key[:-len(".weight")] + ".weight"
                    skey = key[:-len(".weight")] + ".scales"
                    if t.dtype == torch.uint32:
                        sc = _get_scales(src, wm, skey)
                        add(name, dequant(t, sc).to(torch.bfloat16))
                    else:
                        add(name, t.to(torch.bfloat16))
                elif t.dtype == torch.uint32 and name.endswith(".weight"):
                    w8 = t.view(torch.uint8).reshape(t.shape[0], -1)
                    sc = _get_scales(src, wm, skey_for(key, wm))
                    add(name, w8)
                    add(name.replace(".weight", ".weight_scale"),
                        sc.view(torch.float8_e4m3fn))
                    add(name.replace(".weight", ".weight_scale_2"),
                        torch.ones(1, dtype=torch.float32))
                else:
                    add(name, t)

    flush("end")

    index = {
        "metadata": {"total_size": 0},
        "weight_map": {
            k: out_files[i]
            for i, fn in enumerate(out_files)
            for k in _keys_in(dst, fn)
        },
    }
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"wrote {len(out_files)} shards to {dst}")


def skey_for(key: str, wm: dict) -> str:
    return key[:-len(".weight")] + ".scales"


def _get_scales(src: str, wm: dict, skey: str) -> torch.Tensor:
    fn = wm[skey]
    with safe_open(os.path.join(src, fn), "pt") as f:
        return f.get_tensor(skey)


def _keys_in(dst: str, fn: str) -> list[str]:
    from safetensors import safe_open as so

    with so(os.path.join(dst, fn), "pt") as f:
        return list(f.keys())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
