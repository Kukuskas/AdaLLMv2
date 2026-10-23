# PORT PLAN — adallm kernels → vLLM 0.32 plugin

All findings verified against `~/vllm-flash-venv` (vLLM 0.32.0, Python 3.14)
and `~/adallm-nvfp4` (branch dev) on 2026-10-23. File paths below are
relative to the vLLM site-packages unless noted.

## 0. How vLLM 0.32 discovers plugins (exact mechanism)

**Entry points, group `vllm.general_plugins`** — the supported mechanism:

- `vllm/plugins/__init__.py::load_general_plugins()` scans
  `importlib.metadata.entry_points(group="vllm.general_plugins")`, loads
  each entry point, and CALLS the returned callable. Called from
  `vllm/engine/arg_utils.py` (EngineArgs creation), `vllm/v1/engine/core.py`,
  `vllm/v1/worker/worker_base.py`, and the model registry — i.e. in every
  engine-core and worker process. Plugins must be idempotent and
  CUDA-init-free at registration time.
- `VLLM_PLUGINS` env var = name allowlist (unset → load everything).
- Registration seam for quant configs:
  `vllm/model_executor/layers/quantization/__init__.py::register_quantization_config(name)`
  decorator → fills `_CUSTOMIZED_METHOD_TO_QUANT_CONFIG`, appended to
  `QUANTIZATION_METHODS` and the platform's `supported_quantization`. This
  is the sanctioned hook for a custom checkpoint format.
- Registration seam for GEMM kernels (NEW in 0.32, our preferred path):
  `vllm/model_executor/kernels/linear/__init__.py::register_linear_kernel(
  cls, platform, kernel_type="nvfp4")` → appends to
  `_POSSIBLE_NVFP4_KERNELS[platform]`.
- Other groups if needed later: `vllm.logits_processors` (LM-head),
  `vllm.platform_plugins`, `vllm.endpoint_plugins`.

## 1. Where NVFP4 routes on sm89 and why it's slow (the gate)

Checkpoint detection: `hf_quant_config.json` → `ModelConfig._verify_quantization`
(`vllm/config/model.py` ~L1370) → `ModelOptNvFp4Config.override_quantization_method`
(modelopt.py L771) → `"modelopt_fp4"`.

Per-layer dispatch: `ModelOptLinearMethod` → `select_linear_kernel()`
(modelopt.py L2385) → `init_nvfp4_linear_kernel(use_a16=activation is None)`
(`vllm/model_executor/kernels/linear/__init__.py` L1128).

**The sm89 gate chain** (`_POSSIBLE_NVFP4_KERNELS[CUDA]`, in order, each
kernel's `is_supported()`):

| # | kernel | gate on sm89 (cc=89) | outcome |
|---|--------|----------------------|---------|
| 1 | FlashInferCuteDslNvFp4 | `is_device_capability_family(100)` | ✗ |
| 2 | FlashInferCutlassNvFp4 | `cutlass_fp4_supported() && cc>=100` | ✗ (`_C.cutlass_scaled_mm_supports_fp4(89)`=false) |
| 3 | FlashInferB12x | `capability_family(120)` | ✗ |
| 4 | CutlassNvFp4 | same `cutlass_fp4_supported()` | ✗ |
| 5 | FlashInferCuteDsl W4A16 | cc ∈ (100,103,12x) | ✗ |
| 6 | **MarlinNvFp4** | `is_fp4_marlin_supported()` = **cc≥75** | ✓ available |
| 7 | FlashInferTrtllm | `has_flashinfer()` only — **no cc gate** | ✓ wins |
| … | Fbgemm/Torch/B12x | family(100)/12x | ✗ |
| end | Emulation | always | last resort |

Key subtleties:

1. **W4A4 (activations quantized)**: the list is scanned in order; FlashInfer
   TRT-LLM (#7) sits AFTER Marlin (#6), so W4A16 checkpoints DO get Marlin on
   sm89 when `use_a16=True` and auto-select forces Marlin
   (`init_nvfp4_linear_kernel` L1174–1185: `force_kernel = MarlinNvFp4LinearKernel`
   unless `prefers_humming(cc)`, which is only cc=90). But **W4A4** checkpoints
   (NVFP4 quant_algo with input_activations set) skip Marlin (not in
   `a16_kernels` filter — actually it IS scanned, but ordering puts
   FlashInferTrtllm... #6 before #7 means Marlin wins W4A4 too when reached;
   the real issue: **stock Marlin path calls `prepare_fp4_layer_for_marlin`,
   whose `_nvfp4_compute_scale_factor` returns 1.0 for fp16 and zeroes
   adallm-scale-range group scales** — see §2. And for checkpoints whose
   scales hit that bug, the practical fallback observed in prod was the
   TRT-LLM/emulation dequant path at 5.8–6.8 tok/s.)
2. The exact per-kernel capability checks live in
   `vllm/model_executor/kernels/linear/nvfp4/*.py` (`is_supported` methods);
   the Blackwell-only hard gate for CUTLASS FP4 is
   `vllm/_custom_ops.py::cutlass_scaled_mm_supports_fp4` →
   `torch.ops._C.cutlass_scaled_mm_supports_fp4(89)` → false (needs sm100
   tcgen05 mma).
3. `nvfp4_w4a16` auto-select detail (L1174): for W4A16 on cc NOT in
   (100,103) and not 90 → `force_kernel = MarlinNvFp4LinearKernel`. So
   stock vLLM on sm89 already picks Marlin for W4A16 — but with the
   stock `prepare_fp4_layer_for_marlin` scale handling, not adallm's.
   flashinfer 0.7.0 IS installed in the venv, which is what lets TRT-LLM
   win W4A4 layers.

**Conclusion**: the plugin does not need to fight the selector for W4A16 —
it needs to (a) register a kernel that outranks/patches the scale path on
sm89, or (b) simply make `MarlinNvFp4LinearKernel.process_weights_after_loading`
use adallm's repack (our registered kernel with cc==89 gate placed in the
registry achieves this cleanly).

## 2. Kernel → vLLM interface mapping

### 2.1 Marlin NVFP4 GEMM (`adallm/ops/marlin.py`) — THE crown jewel

- **Interface**: `NvFp4LinearKernel` subclass (see
  `kernels/linear/nvfp4/base.py`, 76 lines — 4 methods). Registered via
  `register_linear_kernel(..., kernel_type="nvfp4")` from the
  `vllm.general_plugins` entry point. Consumed by
  `ModelOptNvFp4Config`'s linear method through `init_nvfp4_linear_kernel`.
- **apply_weights** → `apply_fp4_marlin_linear`
  (`model_executor/layers/quantization/utils/marlin_utils_fp4.py` L158) —
  unchanged from adallm's `marlin_linear()` call, except our kernel gets
  `layer.weight / weight_scale / weight_global_scale / workspace` from the
  layer object rather than adallm attribute names.
- **process_weights_after_loading** → vendored `repack_linear()`. The two
  adallm-critical numerics that stock `prepare_fp4_layer_for_marlin`
  handles differently:
  1. **explicit scale_factor**: `sf = 2^ceil(log2(2/(smin*128)))` so that
     `smin*sf*128 ≥ 2` (else `nvfp4_marlin_process_scales` zeroes every
     scale; adallm group scales ~2e-3–2e-2). Stock code path auto-computes
     `_nvfp4_compute_scale_factor` which returns 1.0 for fp16 activations
     → zeroed scales → this is the adallm bugfix that must be preserved.
  2. **weight_scale_2 folding**: per-tensor global scale multiplied into
     group scales, global set to 1.0/sf. Stock path keeps amax/2688
     semantics (KNvfp4Static.process L2061: "Marlin/cutlass want
     ModelOpt's amax/2688"). If the checkpoint is true ModelOpt format the
     stock global-scale handling is already right and only the scale_factor
     fix matters.
- **Checkpoint loading**: ModelOpt NVFP4 layout == adallm QAT layout
  (`uint8 [N, K/2] packed`, `fp8e4m3 [N, K/16] group scales`, fp32
  `weight_scale_2` global) — vLLM's `KNvfp4Static.create_weights`
  (modelopt.py L1987) registers exactly these via
  `ModelWeightParameter`/`PerTensorScaleParameter` with
  TP-aware `input_dim/output_dim`. **No custom weight_loader needed for
  ModelOpt checkpoints.** adallm's own `infra/loader.py` (QAT packed-scale
  format) is only needed if we also serve native adallm checkpoints — that
  becomes a `register_quantization_config("adallm_nvfp4")` QuantConfig with
  its own LinearMethod (phase 3, optional).
- **dtype note**: Marlin needs REAL fp16 activations (bf16 input → 1e-34
  garbage; adallm marlin_linear L274-278 converts bf16→fp16→bf16 around the
  call). Our kernel's apply_weights must do the same. `get_supported_act_dtypes`
  stays [bf16, fp16] — conversion is inside the kernel.
- **Effort**: ~2–3 days (kernel body is mostly done in vendored code;
  remaining work is wiring layer attribute names + tests on GPU).

### 2.2 tiny-N dequant / TINYGEMM path (`marlin.py::_dequant_tiny*`)

- Maps to the same `NvFp4LinearKernel` (or its `can_implement` returning
  False for N < threshold so stock picks another). Cleanest: implement
  inside our kernel's `process_weights_after_loading` — dequant tiny-N
  layers to fp16 weight and route apply_weights to `F.linear` (in-graph
  safe, 20× at M=1). Host-side decision at load; no runtime cost.
- VRAM: +0.4–0.9 GB (27B) for dequantized tiny weights; frees the packed
  copy (net roughly neutral vs packed+Marlin).
- **Effort**: ~1 day (self-contained, pure torch).

### 2.3 splitK fp8-KV attention decode (`adallm/ops/attention.py`)

- **Feasibility: MEDIUM-HARD.** This is a Triton flash-decode over paged
  fp8 KV with adaptive split count derived on-device from context_lens
  (CUDA-graph safe). vLLM interface options:
  a. **Attention backend** (`v1/attention/backends/*.py`, registered in
     `PlatformEnum`/`worker` selection): full contract = plan/metadatas/
     forward + paged KV block manager interplay. The adallm kernel
     assumes adallm's own block_tables layout and fp8-e4m3 KV stored
     dequant-scale-free. Porting to vLLM's paged layout (block_size,
     slot_mapping, kv_cache tensor shape [num_blocks, block, H, D]) is
     real work: ~1–2 weeks.
  b. **Layer-level replacement** is NOT viable in vLLM — attention is
     backend-dispatched, not model-owned, in V1.
  - fp8 KV cache in vLLM 0.32: `ModelOptKVCacheMethod` handles
    kv_cache_quant_algo; the fp8 KV tensor format and per-block scales
    differ from adallm's (which stores k_scale/v_scale folded). Must map
    carefully or keep bf16 KV for v1 of the port.
  - Recommendation: defer to phase 2; ship GEMM first (biggest win, 6→57
    tok/s is dominated by Marlin).
- **Effort**: ~1.5–2 weeks.

### 2.4 LM-head Marlin (`adallm/ops/lmhead_marlin.py`)

- vLLM V1 seam: `vllm.logits_processors` entry-point group
  (`v1/sample/logits_processor/__init__.py` L47) OR a `LogitsProcessor`
  subclass selected when `logits_processor_pattern` matches lm_head. The
  ParallelLMHead would need a Marlin-repacked weight — vLLM quantizes
  lm_head only if the checkpoint includes it in the quant config (it
  usually doesn't). So this requires a custom logits processor that owns
  its own weight repack of `lm_head.weight` post-load.
- **Effort**: ~3–4 days. Lower priority (lm_head GEMM is one layer).

### 2.5 NVFP4 graph warmup (`nvfp4_gemm.py`)

- Not needed as a port: vLLM handles CUDA-graph capture itself. The adallm
  warmup tricks (moving LUTs to device pre-capture, e.g.
  EmulationNvFp4LinearKernel.process_weights_after_loading L28-31 does the
  same) apply automatically since repack happens in
  process_weights_after_loading (outside capture). Only risk: our
  apply_weights must allocate nothing (workspace pre-made at load) —
  satisfied by design.

## 3. VRAM math (27B, RTX 4090 24GB, TP=1)

- Packed NVFP4 weights: 27B params × 0.5 B ≈ 13.5 GB + fp8 scales (K/16 ×
  1 B / param-group) ≈ +0.85 GB → **~14.4 GB** resident.
- Marlin repack is NOT in-place: transient peak during
  `process_weights_after_loading` = packed + marlin-packed for the largest
  layer (~1 GB extra for an 8192×8192 block at fp4+pad). vLLM loads layer
  by layer and frees originals — acceptable, but repack in a loop with
  `torch.cuda.empty_cache()` every N layers like adallm does implicitly.
- Marlin workspace: `sms × 1 int32` = 128 × 4 B — negligible. Shared per
  layer or global.
- tiny-N dequant (if enabled): +0.4–0.9 GB fp16 weights, −(packed tiny
  weights) ≈ net +0.2–0.5 GB.
- Total headroom on 24 GB with 40K-token KV fp8 cache (~4 GB): fits with
  ~4 GB margin. Same envelope as the adallm engine today.

## 4. Risks

1. **CUDA-graph capture** — Marlin GEMM is capture-safe (vLLM already
   captures stock Marlin kernels); our additions (dtype convert bf16↔fp16)
   are pure tensor ops, fine. The splitK attention kernel's adaptive split
   via on-device context_lens was explicitly designed capture-safe — but
   the vLLM backend wrapper must pass real lens tensors, not host ints.
2. **TP sharding interplay** — ModelOpt weight params carry
   input_dim/output_dim so TP slicing of packed uint8 weights is done by
   the loader at fp16-element granularity BEFORE packing-aware slicing.
   Marlin repack then runs per-partition in process_weights_after_loading
   — this is exactly the stock flow, low risk. Merged/fused projections
   (qkv) rely on `output_partition_sizes` list — our create_weights must
   keep that (see scheme stub).
3. **fp8 KV** — adallm's fp8 KV layout (scale-free e4m3, folded scales)
   ≠ vLLM's (`k_scale`/`v_scale` per layer). Phase-2 work; ship GEMM
   first.
4. **Registry ordering** — `register_linear_kernel` appends to the end of
   `_POSSIBLE_NVFP4_KERNELS[CUDA]`; auto-select for W4A16 on sm89 FORCES
   stock Marlin before consulting the registry (L1184-1185), so for W4A16
   we don't even get registry order — our kernel must instead win the
   registry scan (W4A4) or we accept stock-Marlin-forced W4A16 and instead
   patch the scale path via our scheme (register_quantization_config
   "adallm_modelopt_fp4" overriding modelopt_fp4 detection — verified:
   custom methods are checked FIRST in _verify_quantization's override
   loop, L1410-1413). Fallback lever: `VLLM_DISABLED_KERNELS` /
   `--linear-backend` env steering.
5. **vLLM version drift** — `vllm.model_executor.kernels.linear` module is
   new in 0.32; guard imports and no-op on older versions (done in
   plugin.py).

## 5. Phased milestones

- **Phase 0 (this repo, done)**: skeleton, entry point, vendored marlin,
  port plan, CPU-skippable tests.
- **Phase 1 (first GPU milestone, ~3 days)**: fill in
  `AdallmMarlinNvFp4LinearKernel.process_weights_after_loading/apply_weights`
  from vendored code; run `tests/test_marlin_correctness.py` on an idle
  4090: **success = test_marlin_matches_dequant_reference and
  test_marlin_scale_factor_guard pass (rel err < 2e-2), then
  `vllm serve` a small ModelOpt-NVFP4 model (e.g. Nemotron 4B-NVFP4)
  offline-benchmark ≥5× stock tok/s.**
- **Phase 2 (~2 weeks)**: splitK attention backend port.
- **Phase 3 (~1 week)**: lm-head Marlin logits processor; optional
  `adallm_nvfp4` custom quant config for native QAT checkpoints.

## First GPU milestone result (2026-10-23, GPU1, alongside prod)

- `test_cpu_reference_selfconsistent` PASS
- `test_marlin_matches_dequant_reference` FAIL when run directly: rel err 4.3e-2 vs 2e-2 threshold
  (pytest run skipped it due to a skipif evaluation-order bug — marlin_vendored imported before
  CUDA init in the collection context; direct call executes fine, see below)
- Same for `test_marlin_scale_factor_guard` (assert rel < 2e-2 fails; output is NON-zero, so the
  scale_factor zeroing guard itself held — the error is a numerical/convention mismatch, not zeroing)

### Analysis
Reference matmul uses the SAME wq/s as the marlin path, so quant noise cancels — 4.3% is a real
convention difference. Suspects, in order:
1. `_FakeLinear` scale conventions differ from adallm's real Linear (adallm multiplies weight_scale
   by a global scale_factor at load; repack may expect that already applied)
2. marlin kernel internal tile-perm vs the reference's row-major group-of-16 layout in _quantize_nvfp4
3. fp16 accumulate vs fp32 reference

### Next step (milestone M1, blocked on this)
Compare `marlin_vendored.marlin_linear` against adallm's ORIGINAL `ops/marlin.py` call path with the
exact same inputs (import from ~/adallm-nvfp4, run both on GPU, diff outputs). If adallm-original
matches the reference, the vendoring deltas (gptq_marlin_repack signature change!) are the bug.
