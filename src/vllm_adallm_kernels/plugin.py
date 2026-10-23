"""vLLM general-plugin entry point (group ``vllm.general_plugins``).

vLLM 0.32 discovery path (verified against
vllm-flash-venv/lib/python3.14/site-packages/vllm):

  1. ``vllm.plugins.load_general_plugins()`` (called from
     vllm/engine/arg_utils.py, vllm/v1/engine/core.py,
     vllm/v1/worker/worker_base.py, model registry) scans
     ``importlib.metadata.entry_points(group="vllm.general_plugins")``.
  2. Each entry point is ``load()``ed and the returned callable executed
     in EVERY process (engine core + workers), so registration here must
     be idempotent and must not touch CUDA at import time.
  3. ``VLLM_PLUGINS`` env var acts as an allowlist (unset = load all).

What we register (phase 1):
  - ``AdallmMarlinNvFp4LinearKernel`` into the NVFP4 linear-kernel
    registry via ``vllm.model_executor.kernels.linear.register_linear_kernel``
    (kernel_type="nvfp4"). Kernel selection in
    ``init_nvfp4_linear_kernel`` consults the registry in list order; on
    sm89 all native W4A16 kernels above Marlin fail ``is_supported`` or are
    outranked, so ours (is_supported: cc==89) wins for W4A16 and stock
    MarlinNvFp4LinearKernel remains the fallback for other archs.

Phase 2+ (NOT registered yet — see docs/PORT_PLAN.md):
  - ``register_quantization_config("adallm_nvfp4")`` for full-scheme
    checkpoints (QAT layout, packed scales from adallm infra/loader.py).
  - Attention split-K decode backend, LM-head Marlin logits processor.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_REGISTERED = False


def _register_adallm_kernels() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    try:
        from vllm.model_executor.kernels.linear import register_linear_kernel
        from vllm.platforms import PlatformEnum
    except Exception as e:  # vllm missing or layout changed
        logger.warning("vllm-adallm-kernels: vLLM import failed (%s); plugin idle.", e)
        return

    try:
        from .kernel import AdallmMarlinNvFp4LinearKernel
    except Exception as e:
        logger.warning("vllm-adallm-kernels: kernel import failed (%s); plugin idle.", e)
        return

    register_linear_kernel(
        AdallmMarlinNvFp4LinearKernel,
        platform=PlatformEnum.CUDA,
        kernel_type="nvfp4",
    )
    _REGISTERED = True
    logger.info(
        "vllm-adallm-kernels: registered AdallmMarlinNvFp4LinearKernel "
        "for CUDA (nvfp4 registry)."
    )


def register() -> None:
    """Entry-point callable executed by vLLM in every process."""
    _register_adallm_kernels()
