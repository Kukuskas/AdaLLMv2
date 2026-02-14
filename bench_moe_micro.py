import argparse
import json
import os
import time

import torch
from transformers import AutoConfig

from adallm.ops.nvfp4_gemm import nvfp4_linear, nvfp4_linear_dual
from adallm.infra.nvfp4 import get_nvfp4_group_size


def _load_quant_config(path: str) -> dict:
    quant_path = os.path.join(path, "hf_quant_config.json")
    if not os.path.isfile(quant_path):
        return {}
    with open(quant_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_experts(num_experts, hidden, inter, group_size, pin_memory):
    def _alloc(shape, dtype):
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=pin_memory)

    experts = []
    for _ in range(num_experts):
        g_w = _alloc((inter, hidden // 2), torch.uint8)
        g_s = _alloc((inter, hidden // group_size), torch.float16)
        g_s2 = torch.ones([], dtype=torch.float32, device="cpu", pin_memory=pin_memory)
        u_w = _alloc((inter, hidden // 2), torch.uint8)
        u_s = _alloc((inter, hidden // group_size), torch.float16)
        u_s2 = torch.ones([], dtype=torch.float32, device="cpu", pin_memory=pin_memory)
        d_w = _alloc((hidden, inter // 2), torch.uint8)
        d_s = _alloc((hidden, inter // group_size), torch.float16)
        d_s2 = torch.ones([], dtype=torch.float32, device="cpu", pin_memory=pin_memory)
        experts.append((g_w, g_s, g_s2, u_w, u_s, u_s2, d_w, d_s, d_s2))
    return experts


def _to_gpu(expert):
    return tuple(t.to("cuda", non_blocking=True) for t in expert)


def _time_cuda(fn, iters):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / max(iters, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--experts", type=int, default=-1)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--copy-iters", type=int, default=20)
    parser.add_argument("--pin-memory", action="store_true")
    args = parser.parse_args()

    cfg = AutoConfig.from_pretrained(args.model)
    quant_cfg = _load_quant_config(args.model)
    group_size = get_nvfp4_group_size(quant_cfg)
    hidden = cfg.hidden_size
    inter = cfg.moe_intermediate_size
    k = cfg.num_experts_per_tok if args.experts <= 0 else args.experts

    experts_cpu = _make_experts(k, hidden, inter, group_size, args.pin_memory)
    x = torch.randn((args.m, hidden), device="cuda", dtype=torch.float16)

    def copy_only():
        return [_to_gpu(ex) for ex in experts_cpu]

    def compute_only():
        out = 0
        for ex in experts_gpu:
            g_w, g_s, g_s2, u_w, u_s, u_s2, d_w, d_s, d_s2 = ex
            gate, up = nvfp4_linear_dual(x, g_w, g_s, g_s2, u_w, u_s, u_s2, None, None, group_size)
            act = torch.nn.functional.silu(gate) * up
            out = out + nvfp4_linear(act, d_w, d_s, d_s2, None, None, group_size)
        return out

    def copy_and_compute():
        local_gpu = [_to_gpu(ex) for ex in experts_cpu]
        out = 0
        for ex in local_gpu:
            g_w, g_s, g_s2, u_w, u_s, u_s2, d_w, d_s, d_s2 = ex
            gate, up = nvfp4_linear_dual(x, g_w, g_s, g_s2, u_w, u_s, u_s2, None, None, group_size)
            act = torch.nn.functional.silu(gate) * up
            out = out + nvfp4_linear(act, d_w, d_s, d_s2, None, None, group_size)
        return out

    experts_gpu = copy_only()
    torch.cuda.synchronize()
    with torch.inference_mode():
        compute_only()
        torch.cuda.synchronize()
        copy_only()
        torch.cuda.synchronize()

    with torch.inference_mode():
        copy_ms = _time_cuda(copy_only, args.copy_iters)
        compute_ms = _time_cuda(compute_only, args.iters)
        total_ms = _time_cuda(copy_and_compute, max(1, args.copy_iters // 2))

    expert_bytes = (
        inter * (hidden // 2)
        + inter * (hidden // 2)
        + hidden * (inter // 2)
    )
    expert_bytes += (
        inter * (hidden // group_size)
        + inter * (hidden // group_size)
        + hidden * (inter // group_size)
    ) * 2
    expert_bytes += 12
    h2d_mb = (expert_bytes * k) / (1024 ** 2)

    print("MODEL:", args.model)
    print(f"hidden={hidden} inter={inter} group_size={group_size} experts={k} M={args.m}")
    print(f"est_h2d_mb_per_layer={h2d_mb:.2f}")
    print(f"copy_ms={copy_ms:.3f} compute_ms={compute_ms:.3f} total_ms={total_ms:.3f}")


if __name__ == "__main__":
    main()
