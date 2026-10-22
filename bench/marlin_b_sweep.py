#!/usr/bin/env python
"""Marlin NVFP4 vs Triton decode-shape sweep (agent-B, GPU1).

Modes (run as separate processes so module-level env vars take effect):
  triton  — bench adallm Triton nvfp4_linear (NVFP4_FP8 env decides fp16/fp8-act path)
  marlin  — bench vLLM apply_fp4_marlin_linear + correctness vs fp32 reference
            + workspace-sharing test + fp8-act probe
  graph   — CUDA graph capture/replay test for marlin (and triton for comparison)

Env (always): CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
"""
import argparse
import importlib.util
import json
import os
import sys

import torch

REPO = "/home/kukuskas/adallm-marlin-b"
NVFP4_GEMM_PATH = os.path.join(REPO, "adallm/ops/nvfp4_gemm.py")

# (name, K, N) — real decode shapes of Qwen3.8-27B NVFP4
# full-attn (16 layers): q_proj, k/v proj, o_proj
# delta-net (48 layers): in_proj_qkv, in_proj_z(=q_proj shape), in_proj_a/b, out_proj
# all 64 layers: gate/up (N=17408 each), down
# lm_head
SHAPES = [
    ("gate_up", 5120, 17408),
    ("down", 17408, 5120),
    ("q_proj_and_in_proj_z", 5120, 6144),
    ("kv_proj", 5120, 1024),
    ("o_proj_and_out_proj", 6144, 5120),
    ("in_proj_qkv", 5120, 10240),
    ("in_proj_a_b", 5120, 48),
    ("lm_head", 5120, 248320),
]
MS = [1, 2, 3, 4, 5, 6, 7, 8]

FP4_VALS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def load_triton_module(fp8: int):
    os.environ["NVFP4_FP8"] = str(fp8)
    spec = importlib.util.spec_from_file_location(f"nvfp4_gemm_fp8{fp8}", NVFP4_GEMM_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def quant_pack(W, group=16):
    """Quantize fp16 W [N,K] to adallm NVFP4 layout. Returns (packed uint8 [N,K/2],
    scales fp16 [N,K/16], dequant reference fp32)."""
    N, K = W.shape
    amax = W.abs().view(N, K // group, group).amax(-1)
    scale = (amax / 6.0).clamp(max=448.0).half()
    scale[scale == 0] = 1.0
    Wq = (W.view(N, K // group, group) / scale[..., None]).view(N, K)
    vals = torch.tensor(FP4_VALS, device=W.device, dtype=torch.float32)
    codes = torch.empty(N, K, device=W.device, dtype=torch.uint8)
    deq = torch.empty(N, K, device=W.device, dtype=torch.float32)
    step = max(1, (1 << 26) // K)  # keep [chunk,K,8] under ~512MB fp32
    for i in range(0, N, step):
        mag = Wq[i:i + step].float().abs().clamp(max=6.0)
        idx = (mag[..., None] - vals).abs().argmin(-1)  # [chunk,K] 0..7
        sign = (Wq[i:i + step] < 0).to(torch.uint8)
        codes[i:i + step] = idx.to(torch.uint8) | (sign << 3)
        deq[i:i + step] = vals[idx] * torch.sign(Wq[i:i + step]).float() * \
            scale.repeat_interleave(group, dim=1)[i:i + step].float()
    code = codes.view(N, K // 2, 2)
    packed = (code[..., 0] | (code[..., 1] << 4)).contiguous()
    return packed, scale.contiguous(), deq


def bench_us(fn, iters=100, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters


def gbps(K, N, M, us):
    bytes_ = N * K / 2 + N * K / 16 * 2 + M * K * 2 + M * N * 2
    return bytes_ / (us * 1e-6) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["triton", "marlin", "graph"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    assert torch.cuda.is_available()
    dev = torch.cuda.current_device()
    print(f"device {dev}: {torch.cuda.get_device_name(dev)}", flush=True)
    res = {"mode": args.mode, "gpu": torch.cuda.get_device_name(dev), "rows": []}

    if args.mode == "triton":
        fp8 = int(os.environ.get("NVFP4_FP8", "0"))
        mod = load_triton_module(fp8)
        res["triton_fp8"] = fp8
        gen = torch.Generator(device="cuda").manual_seed(1234)
        for name, K, N in SHAPES:
            W = torch.randn(N, K, device="cuda", dtype=torch.float16, generator=gen) * 0.5
            packed, scale, _ = quant_pack(W)
            del W
            s2 = torch.tensor([1.0], device="cuda", dtype=torch.float32)
            for M in MS:
                x = torch.randn(M, K, device="cuda", dtype=torch.float16, generator=gen)
                us = bench_us(
                    lambda: mod.nvfp4_linear(x, packed, scale, s2, None, None, 16),
                    iters=args.iters,
                )
                res["rows"].append(
                    dict(shape=name, K=K, N=N, M=M, kernel=f"triton_fp{8 if fp8 else 16}", us=round(us, 2),
                         gbps=round(gbps(K, N, M, us), 1))
                )
                print(res["rows"][-1], flush=True)
            del packed, scale
            torch.cuda.empty_cache()

    elif args.mode == "marlin":
        import vllm._custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_make_workspace_new, marlin_pad_qweight, marlin_pad_scales,
            marlin_permute_scales, marlin_padded_nk,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            nvfp4_marlin_process_scales, nvfp4_marlin_process_global_scale,
            apply_fp4_marlin_linear,
        )

        gen = torch.Generator(device="cuda").manual_seed(1234)
        workspace = marlin_make_workspace_new(torch.device("cuda"))

        def make_marlin(packed, scale, N, K):
            padded_n, padded_k = marlin_padded_nk(N, K, 16)
            qw = packed.view(torch.int32).T.contiguous()
            qw = marlin_pad_qweight(qw, N, K, padded_n, padded_k)
            mqw = ops.gptq_marlin_repack(
                b_q_weight=qw, perm=torch.empty(0, dtype=torch.int, device="cuda"),
                size_k=padded_k, size_n=padded_n, num_bits=4, is_a_8bit=False)
            ws = scale.T.contiguous()
            ws = marlin_pad_scales(ws, N, K, padded_n, padded_k, 16)
            ws = marlin_permute_scales(ws, padded_k, padded_n, 16, is_a_8bit=False)
            ws, sf = nvfp4_marlin_process_scales(ws, a_dtype=torch.float16)
            gs = nvfp4_marlin_process_global_scale(
                torch.tensor(1.0, device="cuda", dtype=torch.float32), torch.float16) / sf
            return mqw, ws, gs, padded_n, padded_k

        # also load triton fp16 for cross-check
        tmod = load_triton_module(0)

        for name, K, N in SHAPES:
            W = torch.randn(N, K, device="cuda", dtype=torch.float16, generator=gen) * 0.5
            packed, scale, deq = quant_pack(W)
            del W
            mqw, mws, mgs, pn, pk = make_marlin(packed, scale, N, K)
            for M in MS:
                x = torch.randn(M, K, device="cuda", dtype=torch.float16, generator=gen)
                out = apply_fp4_marlin_linear(x, mqw, mws, mgs, workspace, N, K)
                # correctness vs fp32 dequant reference (chunked for lm_head)
                if K * N <= 17408 * 5120 * 2:
                    ref = (x.float() @ deq.T)
                else:
                    ref = torch.empty(M, N, device="cuda", dtype=torch.float32)
                    step = 8192
                    for i in range(0, N, step):
                        ref[:, i:i + step] = x.float() @ deq[i:i + step].T
                rel = ((out.float() - ref).norm() / ref.norm()).item()
                cs = torch.nn.functional.cosine_similarity(
                    out.float().flatten(), ref.flatten(), dim=0).item()
                us = bench_us(
                    lambda: apply_fp4_marlin_linear(x, mqw, mws, mgs, workspace, N, K),
                    iters=args.iters)
                res["rows"].append(
                    dict(shape=name, K=K, N=N, M=M, kernel="marlin", us=round(us, 2),
                         gbps=round(gbps(K, N, M, us), 1), rel_err=round(rel, 5), cos=round(cs, 6)))
                print(res["rows"][-1], flush=True)
                del out, ref
            del packed, scale, deq, mqw, mws
            torch.cuda.empty_cache()

        # ---- workspace sharing test: one workspace across all shapes, 20 seq passes
        print("== workspace sharing test ==", flush=True)
        ws_ok = {"max_abs_diff": 0.0}
        layers = []
        for name, K, N in SHAPES[:7]:  # skip lm_head for memory
            W = torch.randn(N, K, device="cuda", dtype=torch.float16, generator=gen) * 0.5
            packed, scale, _ = quant_pack(W)
            mqw, mws, mgs, pn, pk = make_marlin(packed, scale, N, K)
            x = torch.randn(1, K, device="cuda", dtype=torch.float16, generator=gen)
            solo_ws = marlin_make_workspace_new(torch.device("cuda"))
            ref_out = apply_fp4_marlin_linear(x, mqw, mws, mgs, solo_ws, N, K).clone()
            layers.append((name, x, mqw, mws, mgs, N, K, ref_out))
            del packed, scale, W
        import time
        t0 = time.time()
        for it in range(20):
            for (name, x, mqw, mws, mgs, N, K, ref_out) in layers:
                o = apply_fp4_marlin_linear(x, mqw, mws, mgs, workspace, N, K)
                d = (o - ref_out).abs().max().item()
                ws_ok["max_abs_diff"] = max(ws_ok["max_abs_diff"], d)
        torch.cuda.synchronize()
        ws_ok["sec"] = round(time.time() - t0, 2)
        ws_ok["workspace_nonzero_after"] = int((workspace != 0).sum().item())
        res["workspace_test"] = ws_ok
        print(ws_ok, flush=True)
        del layers
        torch.cuda.empty_cache()

        # ---- concurrent streams sharing one workspace (race probe)
        name, K, N = "kv_proj", 5120, 1024
        W = torch.randn(N, K, device="cuda", dtype=torch.float16, generator=gen) * 0.5
        packed, scale, _ = quant_pack(W)
        mqw, mws, mgs, pn, pk = make_marlin(packed, scale, N, K)
        x = torch.randn(1, K, device="cuda", dtype=torch.float16, generator=gen)
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()
        cur = torch.cuda.current_stream()
        # serial reference
        ref_a = apply_fp4_marlin_linear(x, mqw, mws, mgs, workspace, N, K).clone()
        ref_b = apply_fp4_marlin_linear(x, mqw, mws, mgs, workspace, N, K).clone()
        max_conc = 0.0
        for _ in range(10):
            with torch.cuda.stream(s1):
                a = apply_fp4_marlin_linear(x, mqw, mws, mgs, workspace, N, K)
            with torch.cuda.stream(s2):
                b = apply_fp4_marlin_linear(x, mqw, mws, mgs, workspace, N, K)
            torch.cuda.synchronize()
            max_conc = max(max_conc, (a - ref_a).abs().max().item(), (b - ref_b).abs().max().item())
        res["stream_race_test"] = {"max_abs_diff_concurrent": max_conc}
        print(res["stream_race_test"], flush=True)

        # ---- fp8 activation probe
        fp8probe = {}
        try:
            apply_fp4_marlin_linear(
                torch.randn(1, K, device="cuda").to(torch.float8_e4m3fn),
                mqw, mws, mgs, workspace, N, K,
                input_dtype=torch.float8_e4m3fn)
            fp8probe["nvfp4_fp8_act"] = "ACCEPTED (unexpected)"
        except RuntimeError as ex:
            fp8probe["nvfp4_fp8_act"] = f"RuntimeError: {ex}"
        # fp16 input with input_dtype flag (MXFP4-style path w/ nvfp4 weights):
        try:
            apply_fp4_marlin_linear(x, mqw, mws, mgs, workspace, N, K,
                                    input_dtype=torch.float8_e4m3fn)
            fp8probe["nvfp4_fp8_flag"] = "ACCEPTED (unexpected)"
        except RuntimeError as ex:
            fp8probe["nvfp4_fp8_flag"] = f"RuntimeError: {ex}"
        res["fp8_probe"] = fp8probe
        print(fp8probe, flush=True)

    elif args.mode == "graph":
        import vllm._custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_make_workspace_new, marlin_pad_qweight, marlin_pad_scales,
            marlin_permute_scales, marlin_padded_nk,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            nvfp4_marlin_process_scales, nvfp4_marlin_process_global_scale,
            apply_fp4_marlin_linear,
        )
        tmod = load_triton_module(0)
        gen = torch.Generator(device="cuda").manual_seed(7)
        res["graphs"] = []
        GRAPH_SHAPES = [("gate_up", 5120, 17408), ("down", 17408, 5120),
                        ("q_proj_and_in_proj_z", 5120, 6144), ("kv_proj", 5120, 1024),
                        ("o_proj_and_out_proj", 6144, 5120)]
        for name, K, N in GRAPH_SHAPES:
            for M in ([1, 4, 8] if name == "gate_up" else [1]):
                W = torch.randn(N, K, device="cuda", dtype=torch.float16, generator=gen) * 0.5
                packed, scale, _ = quant_pack(W)
                padded_n, padded_k = marlin_padded_nk(N, K, 16)
                qw = packed.view(torch.int32).T.contiguous()
                qw = marlin_pad_qweight(qw, N, K, padded_n, padded_k)
                mqw = ops.gptq_marlin_repack(
                    b_q_weight=qw, perm=torch.empty(0, dtype=torch.int, device="cuda"),
                    size_k=padded_k, size_n=padded_n, num_bits=4, is_a_8bit=False)
                ws = marlin_pad_scales(scale.T.contiguous(), N, K, padded_n, padded_k, 16)
                ws = marlin_permute_scales(ws, padded_k, padded_n, 16, is_a_8bit=False)
                ws, sf = nvfp4_marlin_process_scales(ws, a_dtype=torch.float16)
                mgs = nvfp4_marlin_process_global_scale(
                    torch.tensor(1.0, device="cuda", dtype=torch.float32), torch.float16) / sf
                del W, qw
                workspace = marlin_make_workspace_new(torch.device("cuda"))
                static_x = torch.randn(M, K, device="cuda", dtype=torch.float16, generator=gen)
                s2 = torch.tensor([1.0], device="cuda", dtype=torch.float32)

                # eager warmup (also compiles triton kernel)
                for _ in range(3):
                    eager_marlin = apply_fp4_marlin_linear(static_x, mqw, ws, mgs, workspace, N, K)
                    eager_triton = tmod.nvfp4_linear(static_x, packed, scale, s2, None, None, 16)
                torch.cuda.synchronize()

                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    g_out_m = apply_fp4_marlin_linear(static_x, mqw, ws, mgs, workspace, N, K)
                    g_out_t = tmod.nvfp4_linear(static_x, packed, scale, s2, None, None, 16)
                g.replay()
                torch.cuda.synchronize()
                dm = (g_out_m - eager_marlin).abs().max().item()
                dt_ = (g_out_t - eager_triton).abs().max().item()
                # 100 replays, determinism + timing
                first = g_out_m.clone()
                us_g = bench_us(lambda: g.replay(), iters=100, warmup=10)
                torch.cuda.synchronize()
                drep = (g_out_m - first).abs().max().item()
                us_e_m = bench_us(lambda: apply_fp4_marlin_linear(static_x, mqw, ws, mgs, workspace, N, K),
                                  iters=args.iters)
                us_e_t = bench_us(lambda: tmod.nvfp4_linear(static_x, packed, scale, s2, None, None, 16),
                                  iters=args.iters)
                row = dict(shape=name, K=K, N=N, M=M,
                           graph_marlin_vs_eager_maxdiff=dm,
                           graph_triton_vs_eager_maxdiff=dt_,
                           graph_100replay_drift=drep,
                           us_graph_both=round(us_g, 2),
                           us_eager_marlin=round(us_e_m, 2),
                           us_eager_triton_fp16=round(us_e_t, 2))
                res["graphs"].append(row)
                print(row, flush=True)
                del packed, scale, mqw, ws
                torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
