#!/usr/bin/env bash
# adallm-driver: model-manager-go driver wrapper for AdaLLM (NVFP4 on RTX 4090).
# Rewrites mm-go's fixed argv (--model <path> --host H --port P) into:
#   adallm serve <model_path> --host H --port P [extra adallm flags from params]
# Extra params pass through only when safelisted (adallm serve flags).
set -euo pipefail

VENV=/home/kukuskas/vllm-venv
MODEL=""
HOST="127.0.0.1"
PORT=""
EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)  MODEL="$2"; shift 2 ;;
        --host)   HOST="$2"; shift 2 ;;
        --port)   PORT="$2"; shift 2 ;;
        # safelisted adallm serve flags (pass through with their value)
        --max-model-len|--max-num-seqs|--max-num-batched-tokens|\
        --gpu-memory-utilization|--tensor-parallel-size|--kv-cache-dtype|\
        --kv-cache-max-blocks|--fp8-decode-kernel-min-ctx)
            EXTRA+=("$1" "$2"); shift 2 ;;
        --enforce-eager|--no-fp8-decode-kernel-enabled|--fp8-decode-kernel-enabled)
            EXTRA+=("$1"); shift ;;
        # llama.cpp-style params mm-go may inject: ignore (except --reasoning)
        --ctx-size|--ctx|--ngl|--n-gpu-layers|--parallel|--flash-attn)
            shift 2 ;;
        --reasoning)
            # mm-go injects --reasoning on|off for benchmark thinking control.
            # Map to env; serve.py turns it into enable_thinking template kwarg.
            ADALLM_REASONING="$2"; export ADALLM_REASONING; shift 2 ;;
        --tensor-parallel-size)
            # TP>1 requires both GPUs visible (round8-D/E) — set ADALLM_TP so
            # CUDA_VISIBLE_DEVICES picks up both devices below.
            ADALLM_TP="$2"; export ADALLM_TP
            EXTRA+=("--tensor-parallel-size" "$2"); shift 2 ;;
        *) shift ;;
    esac
done

if [[ -z "$MODEL" || -z "$PORT" ]]; then
    echo "adallm-driver: missing --model or --port" >&2
    exit 2
fi

export ADALLM_QUIET=0
export NVFP4_FP8=${NVFP4_FP8:-1}       # FP8 GEMM fast path (Triton mode only)
export ADALLM_MARLIN=${ADALLM_MARLIN:-1}  # Marlin NVFP4 kernel: 22.9 tok/s single-stream (vs 9.8)
export ADALLM_NO_COMPILE=1   # inductor fork-pool deadlocks under server load
export ADALLM_NO_PIN=1       # pinned allocs from engine thread can deadlock
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ "${ADALLM_TP:-}" == "2" ]]; then
    export CUDA_VISIBLE_DEVICES=0,1   # TP=2 (round8-D/E): both GPUs
else
    export CUDA_VISIBLE_DEVICES=0
fi
# Per-model Marlin gate (2026-10-23): compressed-tensors (QUASAR) AND modelopt
# (official NVFP4 export) produce garbage ("!!!!") under the Marlin kernel;
# only the QAT custom format is Marlin-safe. Bisected on GPU1 + confirmed live
# on the official bench server (5804) 2026-10-23 05:40 UTC.
case "$MODEL" in
    *QUASAR*|*/Qwen3.8-27B-NVFP4) export ADALLM_MARLIN=0 ;;
    *)                             export ADALLM_MARLIN=${ADALLM_MARLIN:-1} ;;
esac
# --- round2+3 speed stack (2026-10-22): 9.8 -> 34-37.5 tok/s single, 4-stream 152 agg ---
export ADALLM_ATTN_SPLITK=${ADALLM_ATTN_SPLITK:-1}   # split-K flash decode: 4.8x@16K/9.4x@32K kernel
export ADALLM_TINYGEMM=${ADALLM_TINYGEMM:-1}         # tiny-N dequant + cuBLAS + pair fusion
export ADALLM_FP16_ACT=${ADALLM_FP16_ACT:-1}         # fp16 activations end-to-end (-1380 converts)
export ADALLM_LMHEAD_MARLIN=${ADALLM_LMHEAD_MARLIN:-1}  # lm_head NVFP4+Marlin (98% top-1)
# --- concurrency: queue server + interleaved prefill (4 streams 152 tok/s agg) ---
export ADALLM_QUEUE_SERVER=${ADALLM_QUEUE_SERVER:-1}
export ADALLM_INTERLEAVE=${ADALLM_INTERLEAVE:-2}
export ADALLM_CHUNK_BUDGET=${ADALLM_CHUNK_BUDGET:-4096}
# --- round4-b (2026-10-22): prefill +13-16%, -1.2GB VRAM, graph sampler ---
export ADALLM_DELTA_CHUNK=${ADALLM_DELTA_CHUNK:-1}   # chunked delta-net prefill kernel (4x)
export ADALLM_MLP_NO_CAT=${ADALLM_MLP_NO_CAT:-1}     # silu(g)*u without cat round-trip
export ADALLM_EMBED_FP8=${ADALLM_EMBED_FP8:-1}       # embedding fp8 (2.4->1.19 GiB, byte-identical outputs)
# --- round5-b (2026-10-22): parallel prefill — mixed workloads 78->168 tok/s ---
export ADALLM_PARALLEL_PREFILL=${ADALLM_PARALLEL_PREFILL:-1}
# --- round8 (2026-10-23): MTP speculative decoding, graphed verify+replay.
# OFF by default: single-stream only (spec engine assumes 1 running seq).
# Verified on testbed: verify 185->28.8ms, replay 176->23.6ms, 74 vs 148 ms/tok
# @28K ctx = 2.0x net win, token-identical output. Enable per-model via mm-go
# params (env passthrough) ONLY for single-stream serving profiles.
export ADALLM_MTP=${ADALLM_MTP:-0}
export ADALLM_MTP_GRAPH=${ADALLM_MTP_GRAPH:-1}
export ADALLM_PAGED_VERIFY=${ADALLM_PAGED_VERIFY:-1}
# TP=2 (round8-D/E): off by default — requires mm-go to pass --tensor-parallel-size 2
# and both GPUs. 55.8 tok/s single-stream with delta-net head sharding (99d8935).
exec "$VENV/bin/python" -m adallm.cli serve "$MODEL" --host "$HOST" --port "$PORT" "${EXTRA[@]}"
