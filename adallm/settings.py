import json
import os
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from safetensors import safe_open
from transformers import AutoConfig


def _dtype_from_str(dtype_str: str | None) -> torch.dtype:
    if dtype_str is None:
        return torch.float32
    dtype_str = dtype_str.lower()
    if dtype_str in ("bfloat16", "bf16"):
        return torch.bfloat16
    if dtype_str in ("float16", "fp16", "half"):
        return torch.float16
    if dtype_str in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_str}")


def _infer_gemma3_text_params(model_path: str) -> dict:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"Missing safetensors index: {index_path}")
    with open(index_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    def _shape(name: str) -> tuple[int, ...]:
        file = weight_map[name]
        with safe_open(os.path.join(model_path, file), framework="pt", device="cpu") as sf:
            return tuple(sf.get_tensor(name).shape)

    head_dim = _shape("language_model.model.layers.0.self_attn.q_norm.weight")[0]
    q_out, hidden_size = _shape("language_model.model.layers.0.self_attn.q_proj.weight")
    k_out, _ = _shape("language_model.model.layers.0.self_attn.k_proj.weight")
    vocab_size, _ = _shape("language_model.model.embed_tokens.weight")
    num_attention_heads = q_out // head_dim
    num_key_value_heads = k_out // head_dim
    return dict(
        head_dim=head_dim,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
    )


def _load_gemma3_config(model_path: str):
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model_type = cfg.get("model_type", "")
    if model_type not in ("gemma3", "gemma3_text"):
        raise ValueError(f"Unsupported model_type: {model_type}")

    text_cfg = cfg.get("text_config", {})
    inferred = _infer_gemma3_text_params(model_path)
    hidden_size = text_cfg.get("hidden_size", inferred["hidden_size"])
    num_attention_heads = text_cfg.get("num_attention_heads", inferred["num_attention_heads"])
    num_key_value_heads = text_cfg.get("num_key_value_heads", inferred["num_key_value_heads"])
    head_dim = text_cfg.get("head_dim", inferred["head_dim"])
    vocab_size = text_cfg.get("vocab_size", inferred["vocab_size"])

    torch_dtype = _dtype_from_str(cfg.get("torch_dtype") or cfg.get("dtype"))
    hidden_activation = text_cfg.get("hidden_activation", text_cfg.get("hidden_act", "gelu_pytorch_tanh"))
    query_pre_attn_scalar = text_cfg.get("query_pre_attn_scalar", hidden_size // num_attention_heads)
    max_position_embeddings = text_cfg.get("max_position_embeddings", 131072)

    return SimpleNamespace(
        model_type=text_cfg.get("model_type", "gemma3_text"),
        architectures=cfg.get("architectures", ["Gemma3ForConditionalGeneration"]),
        hidden_size=hidden_size,
        intermediate_size=text_cfg.get("intermediate_size", None),
        num_hidden_layers=text_cfg.get("num_hidden_layers", None),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        vocab_size=vocab_size,
        rms_norm_eps=text_cfg.get("rms_norm_eps", 1e-6),
        attention_bias=text_cfg.get("attention_bias", False),
        hidden_act=hidden_activation,
        hidden_activation=hidden_activation,
        query_pre_attn_scalar=query_pre_attn_scalar,
        rope_theta=text_cfg.get("rope_theta", 1000000.0),
        rope_local_base_freq=text_cfg.get("rope_local_base_freq", None),
        rope_scaling=text_cfg.get("rope_scaling", None),
        max_position_embeddings=max_position_embeddings,
        sliding_window=text_cfg.get("sliding_window", None),
        layer_types=text_cfg.get("layer_types", None),
        tie_word_embeddings=cfg.get("tie_word_embeddings", True),
        torch_dtype=torch_dtype,
        quantization_config=cfg.get("quantization_config", None),
    )


def _load_local_config(model_path: str):
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model_type = cfg.get("model_type", "")
    if model_type in ("gemma3", "gemma3_text"):
        return _load_gemma3_config(model_path)
    return None


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    nvfp4_mode: str = "kernel"
    kv_cache_dtype: str = "fp8"
    hybrid_kv_cache: bool = False
    kv_cache_sink_blocks: int = 0
    kv_cache_recent_blocks: int = 0
    warmup_tokens: int = 0
    warmup_num_seqs: int = 1
    kv_cache_max_blocks: int = -1
    fp8_decode_kernel_enabled: bool = True
    fp8_decode_kernel_min_ctx: int = 0
    moe_offload_experts: bool = True
    moe_gpu_expert_capacity: int = 8
    moe_offload_pin_memory: bool = True
    moe_offload_prefetch: bool = True
    moe_offload_prefetch_window: int = 3
    moe_offload_pack: bool = True
    moe_offload_pack_capacity: int = 0
    moe_offload_pack_cache: bool = True
    moe_offload_pack_reuse: bool = True
    moe_offload_pack_gather_min_rows: int = 128
    moe_cache_decay: float = 0.95
    moe_cache_decay_interval: int = 128
    moe_profile: bool = False
    moe_profile_interval: int = 1
    moe_profile_layers: tuple[int, ...] | None = None

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = _load_local_config(self.model)
        if self.hf_config is None:
            self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
        if self.kv_cache_max_blocks < 0:
            # Cap KV cache to configured max tokens instead of filling GPU memory.
            self.kv_cache_max_blocks = max(
                1,
                (self.max_num_batched_tokens + self.kvcache_block_size - 1) // self.kvcache_block_size,
            )
        if self.moe_profile_layers is not None:
            if isinstance(self.moe_profile_layers, int):
                self.moe_profile_layers = (self.moe_profile_layers,)
            elif isinstance(self.moe_profile_layers, list):
                self.moe_profile_layers = tuple(self.moe_profile_layers)
