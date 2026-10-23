"""vLLM general-plugin entry point (group ``vllm.general_plugins``).

vLLM 0.32 discovery path (verified against
vllm-flash-venv/lib/python3.14/site-packages/vllm):

  1. ``vllm.plugins.load_general_plugins()`` (called from
     ``vllm/engine/arg_utils.py``, ``vllm/v1/engine/core.py``,
     ``vllm/v1/worker/worker_base.py``, model registry) scans
     ``importlib.metadata.entry_points(group="vllm.general_plugins")``.
  2. Each entry point is ``load()``ed and the returned callable executed
     in EVERY process (engine core + workers), so registration here must
     be idempotent and must not touch CUDA at import time.
  3. ``VLLM_PLUGINS`` env var acts as an allowlist (unset = load all).

What we register:
  1. ``AdallmMarlinNvFp4LinearKernel`` into the NVFP4 linear-kernel
     registry (``register_linear_kernel(..., kernel_type="nvfp4")``).
     Registry order decides W4A4 ModelOpt layers on sm89 (ours claims
     cc==89 before the cc-gateless FlashInfer TRT-LLM entry).
  2. Custom quantization configs ``adallm_nvfp4`` (adallm QAT
     checkpoints, pre-converted layout) and ``adallm_modelopt_fp4``
     (detection steal for stock ModelOpt NVFP4 checkpoints) via
     ``register_quantization_config`` — custom methods are scanned BEFORE
     stock overrides in ``ModelConfig._verify_quantization``, so stock
     modelopt_fp4 checkpoints route to our scale-fixed Marlin path without
     touching vLLM.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_REGISTERED = False


def _register_adallm_kernels() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    registered_any = False

    # 1. linear-kernel registry (vLLM >= 0.32 kernels/linear layout)
    try:
        from vllm.model_executor.kernels.linear import register_linear_kernel
        from vllm.platforms import PlatformEnum

        from .kernel import AdallmMarlinNvFp4LinearKernel

        register_linear_kernel(
            AdallmMarlinNvFp4LinearKernel,
            platform=PlatformEnum.CUDA,
            kernel_type="nvfp4",
        )
        registered_any = True
        logger.info(
            "vllm-adallm-kernels: registered AdallmMarlinNvFp4LinearKernel "
            "for CUDA (nvfp4 registry)."
        )
    except Exception as e:
        logger.warning(
            "vllm-adallm-kernels: linear-kernel registration skipped (%s).", e
        )

    # 2. custom quantization configs (checkpoint detection steal)
    try:
        from . import scheme  # noqa: F401  (registers on import, idempotent)
        registered_any = True
        logger.info(
            "vllm-adallm-kernels: registered quantization configs "
            "adallm_nvfp4 + adallm_modelopt_fp4."
        )
    except Exception as e:
        logger.warning(
            "vllm-adallm-kernels: quantization-config registration skipped (%s).", e
        )

    _REGISTERED = True
    if not registered_any:
        logger.warning("vllm-adallm-kernels: nothing registered; plugin idle.")


def register() -> None:
    """Entry-point callable executed by vLLM in every process."""
    _register_adallm_kernels()
