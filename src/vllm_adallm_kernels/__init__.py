"""vllm-adallm-kernels: adallm NVFP4 kernel stack as a vLLM 0.32+ plugin.

Mission: stock vLLM 0.32 on RTX 4090 (sm89) runs NVFP4 checkpoints via a
slow dequant fallback (~6 tok/s eager on the 27B). adallm's Marlin-based
stack reaches ~57 tok/s on the same GPU. This plugin extracts adallm's
kernel stack (Marlin NVFP4 GEMM, split-K fp8-KV attention decode, LM-head
Marlin, tiny-N dequant) into an installable vLLM plugin so ANY
vLLM-supported model gets that speed on Ada GPUs.

Plugin entry point (group ``vllm.general_plugins``): :mod:`.plugin`,
registered in pyproject.toml. Loading is triggered by vLLM itself in every
process (worker / engine core / arg parsing).
"""

__version__ = "0.1.0"

from .scheme import AdallmMarlinScheme  # noqa: F401  (re-export for tests)
