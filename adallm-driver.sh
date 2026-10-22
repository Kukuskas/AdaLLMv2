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
        # llama.cpp-style params mm-go may inject: ignore
        --ctx-size|--ctx|--ngl|--n-gpu-layers|--parallel|--flash-attn|--reasoning)
            shift 2 ;;
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
export CUDA_VISIBLE_DEVICES=0
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
exec "$VENV/bin/python" -m adallm.cli serve "$MODEL" --host "$HOST" --port "$PORT" "${EXTRA[@]}"
