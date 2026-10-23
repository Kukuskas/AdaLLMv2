# vllm-adallm-kernels

adallm's NVFP4 kernel stack as a **vLLM 0.32+ plugin** for Ada (sm89) GPUs.

## Mission

Stock vLLM 0.32 on RTX 4090 runs NVFP4 checkpoints through a dequant
fallback (measured 5.8–6.8 tok/s eager on Flash-Next, 27B model). The
adallm stack (Marlin NVFP4 GEMM + splitK fp8-KV decode + LM-head Marlin +
tiny-N dequant) reaches **57 tok/s** on the same GPU and model. This repo
extracts that stack into an installable plugin so **any** vLLM-supported
model with an NVFP4 (ModelOpt) checkpoint gets the adallm speed on Ada —
no vLLM fork required.

## Status

- [x] vLLM 0.32 plugin surfaces studied (registration, kernel selection,
      sm89 gate) — see `docs/PORT_PLAN.md`
- [x] Plugin skeleton: entry point (`vllm.general_plugins`), kernel
      registry hook, scheme stub
- [x] adallm `marlin.py` vendored (import-adapted only)
- [ ] Phase 1: `AdallmMarlinNvFp4LinearKernel` bodies (GPU)
- [ ] Phase 2: splitK attention backend
- [ ] Phase 3: LM-head Marlin logits processor

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│ vLLM 0.32 process (engine core / workers)                          │
│                                                                    │
│  HF checkpoint w/ hf_quant_config.json (quant_algo: NVFP4)         │
│        │  ModelOptNvFp4Config.override_quantization_method         │
│        ▼                                                           │
│  quantization = "modelopt_fp4"                                     │
│        │  ModelOptLinearMethod → select_linear_kernel()            │
│        ▼                                                           │
│  init_nvfp4_linear_kernel(use_a16)                                 │
│        │  iterates _POSSIBLE_NVFP4_KERNELS[CUDA] in order,         │
│        │  first kernel with is_supported()+can_implement() wins    │
│        ▼                                                           │
│  ┌── THIS PLUGIN ───────────────────────────────────────────┐      │
│  │ AdallmMarlinNvFp4LinearKernel  (cc==89 only)             │      │
│  │   process_weights_after_loading:                          │      │
│  │     vendored marlin_vendored.repack_linear                │      │
│  │       (explicit scale_factor, weight_scale_2 folding)     │      │
│  │   apply_weights:                                          │      │
│  │     apply_fp4_marlin_linear(x.fp16, …)  [W4A16]           │      │
│  └───────────────────────────────────────────────────────────┘      │
│        │ fallback (non-sm89): stock MarlinNvFp4LinearKernel        │
│        │ fallback (sm89, no plugin): FlashInfer TRT-LLM dequant    │
│                                                                    │
│  phases 2–3: splitK fp8-KV decode attention backend;               │
│              LM-head Marlin via vllm.logits_processors             │
└────────────────────────────────────────────────────────────────────┘
```

Host vs device split: checkpoint loading + Marlin repack run once at load
time on device (post-load hook); per-token GEMM is the Marlin kernel;
scale math (`scale_factor`, global-scale division) is host-side Python at
repack time only.

## Install (dev)

```bash
uv pip install -e ".[vllm,test]"   # into a venv that already has vllm 0.32
```

The plugin is discovered via the `vllm.general_plugins` entry point; set
`VLLM_PLUGINS=adallm_kernels` to restrict loading to this plugin only.

## Repo layout

```
src/vllm_adallm_kernels/
  __init__.py        package root; exports AdallmMarlinScheme
  plugin.py          vLLM entry point; kernel-registry registration
  kernel.py          AdallmMarlinNvFp4LinearKernel stub (sm89-gated)
  scheme.py          AdallmMarlinScheme stub (scheme-level port, phase 2)
  marlin_vendored.py adallm ops/marlin.py, import-adapted (see header)
docs/PORT_PLAN.md    kernel→interface mapping, VRAM math, risks
tests/test_marlin_correctness.py  GPU-optional correctness test
```

## License

Apache-2.0 (matches vLLM; adallm code © its authors).
