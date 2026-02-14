from transformers import Qwen3MoeConfig

from adallm.architectures.gemma3 import Gemma3ForConditionalGeneration
from adallm.architectures.qwen3 import Qwen3ForCausalLM, Qwen3MoeForCausalLM, Qwen3NextForCausalLM


def build_model(hf_config):
    model_type = getattr(hf_config, "model_type", "")
    if model_type in ("gemma3", "gemma3_text"):
        return Gemma3ForConditionalGeneration(hf_config), False

    is_next = model_type == "qwen3_next"
    is_moe = isinstance(hf_config, Qwen3MoeConfig) or model_type in ("qwen3_moe", "qwen3_next")
    if is_moe:
        model = Qwen3NextForCausalLM(hf_config) if is_next else Qwen3MoeForCausalLM(hf_config)
    else:
        model = Qwen3ForCausalLM(hf_config)
    return model, is_moe
