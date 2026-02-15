# AdaLLM: an inference engine that enables NVFP4 on RTX-4000 series 


AdaLLM is an NVFP4-first inference runtime for Ada Lovelace (e.g. RTX 4090) with FP8 KV cache and custom decode kernels, cutting peak VRAM usage by ~240% in benchmarks with only ~20-25% throughput loss.
This repo targets NVFP4 weights and keeps the entire decode path in FP8, including Gemma3 sliding-window attention.

## What this project does

- Runs NVFP4-quantized weights on Ada Lovelace GPUs.
- Stores KV cache in FP8 and decodes with a custom FP8 kernel.
- Supports Qwen3 (dense + MoE) and Gemma3 text (including sliding-window layers).
- Keeps an NVFP4-only fast path; no silent fallback to FP16 for decode.
- MoE support works but is not fully optimized yet (in progress).

## Demo
![short_demo](https://github.com/user-attachments/assets/72f4914e-0f0c-401e-ad7a-cd9ebb9ad1f3)


## Quickstart

Install and launch an OpenAI-compatible server:

```bash
pip install git+https://github.com/BenChaliah/NVFP4-on-4090-vLLM.git
adallm serve nvidia/Qwen3-8B-NVFP4
```

You can pass a Hugging Face repo ID like above (it will use the cache if present,
otherwise it will download into the default cache directory) or specify the path of the model directory.

```bash
adallm serve ./models/Qwen3-8B-NVFP4
```

Interactive mode (prompt repeatedly, `\\q` to quit):

```bash
adallm run nvidia/Qwen3-8B-NVFP4
```

Stream to terminal:

```bash
adallm run nvidia/Qwen3-8B-NVFP4 "Once upon a time"
```

## CLI options

`adallm serve <model>` options:

- `--host`, `--port`: Bind address and port (default `0.0.0.0:8000`).
- `--download-dir`: Custom Hugging Face cache directory.
- `--max-model-len`, `--max-num-seqs`, `--max-num-batched-tokens`: Scheduler limits.
- `--gpu-memory-utilization`: Target VRAM utilization for KV cache sizing.
- `--tensor-parallel-size`: Tensor parallel degree.
- `--enforce-eager`: Disable CUDA graphs.
- `--kv-cache-dtype`: `fp8` (default), `fp16`, or `bf16`.
- `--fp8-decode-kernel-enabled` / `--no-fp8-decode-kernel-enabled`: Toggle FP8 decode kernel.
- `--fp8-decode-kernel-min-ctx`: Minimum context length to use the FP8 kernel.

`adallm run <model> [prompt]` options:

- All `serve` options above for model/runtime control.
- `--temperature`, `--max-tokens`, `--ignore-eos`, `--seed`: Sampling controls.
- `--chat` / `--no-chat`: Force chat template and history in interactive mode.

Environment toggles:

- `ADALLM_QUIET=1`: Suppress startup logs.
- `ADALLM_DISABLE_PROGRESS=1`: Hide weight-loading progress bar.

## Models

Tested with locally downloaded weights:

- Qwen3-8B-NVFP4  
  https://huggingface.co/nvidia/Qwen3-8B-NVFP4
- Gemma3-27B-it-NVFP4  
  https://huggingface.co/BenChaliah/Gemma3-27B-it-NVFP4

## How it works (short version)

1. `adallm.runtime.LLMEngine` builds a scheduler and a model runner.
2. NVFP4 weights are loaded via `adallm.infra.loader`, preserving scales in kernel mode.
3. KV cache is allocated in FP8 and attached to attention layers.
4. Prefill uses FlashAttention (varlen); decode uses a custom FP8 kernel.
5. Gemma3 sliding-window attention is honored in FP8 decode (no FP16 fallback).

Key modules:

- `adallm/runtime`: process orchestration, scheduling, and graph capture.
- `adallm/architectures`: model definitions + registry.
- `adallm/ops`: NVFP4 linear, attention, rotary, and sampler kernels.
- `adallm/infra`: NVFP4 quant loader and runtime helpers.

## Benchmarks (RTX 4090, Ada Lovelace)

All runs:

- `max_tokens=128`
- prompt: `"Once upon a time"`
- `kv_cache_dtype=fp8`
- `fp8_decode_kernel_enabled=True`
- `fp8_decode_kernel_min_ctx=0`
- measured after warmup

Qwen3-8B-NVFP4

| batch | total tokens | seconds | tok/s | peak GB |
|------:|-------------:|--------:|------:|--------:|
| 1 | 128 | 3.3867 | 37.79 | 7.55 |
| 2 | 256 | 3.5471 | 72.17 | 7.55 |
| 4 | 512 | 3.4392 | 148.87 | 7.55 |
| 8 | 1024 | 3.4459 | 297.16 | 7.56 |
| 16 | 2048 | 4.3636 | 469.34 | 7.56 |

Gemma3-27B-it-NVFP4

| batch | total tokens | seconds | tok/s | peak GB |
|------:|-------------:|--------:|------:|--------:|
| 1 | 128 | 9.3982 | 13.62 | 19.83 |
| 2 | 256 | 9.5545 | 26.79 | 19.83 |
| 4 | 512 | 9.5344 | 53.70 | 19.84 |

Re-run benchmarks:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 NVFP4_FP8=1 \
python scripts/bench_nvfp4.py \
  --model-path ../Qwen3-8B-NVFP4 \
  --batch-sizes 1,2,4,8,16 \
  --max-tokens 128 \
  --seed 7

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 NVFP4_FP8=1 \
python scripts/bench_nvfp4.py \
  --model-path ../Gemma3-27B-it-NVFP4 \
  --batch-sizes 1,2,4 \
  --max-tokens 128
```

To avoid prefix-cache effects in multi-batch runs, you can use a distinct prompt set:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 NVFP4_FP8=1 \
python scripts/bench_nvfp4.py \
  --model-path ../Qwen3-8B-NVFP4 \
  --batch-sizes 1,2,4,8,16 \
  --max-tokens 128 \
  --seed 7 \
  --use-prompt-set
```

You can also pass your own prompt list with `--prompts-file`.

## NVFP4 on Ada Lovelace

This runtime assumes:

- NVIDIA Ada Lovelace GPU (e.g., RTX 4090)
- NVFP4 weight format
- FP8 KV cache and custom decode kernel enabled

The combination enables lower VRAM usage and higher decode throughput compared to FP16 baselines while preserving coherent generation on supported models.

## MoE status

MoE inference is supported for Qwen3 MoE variants, but expert routing/CPU-offload paths are still being optimized. Expect low throughput compared to dense models for now.

MoE smoke test (Once upon a time, max_tokens=128, seed=7):
- Model: `Qwen3-30B-A3B-NVFP4` (weights: `https://huggingface.co/nvidia/Qwen3-30B-A3B-NVFP4`)
- Throughput: ~2.7 tok/s
- Peak VRAM: ~4.67 GB
