"""MTP draft module for Qwen3.8 (EAGLE-style, 1 layer).
Loads language_model.mtp.* weights. Norm convention detected at load.
Draft forward: x = fc(cat(norm_h(h), norm_e(embed(tok)))) -> 1 full-attn layer -> norm -> lm_head
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


class MTPDraft(nn.Module):
    def __init__(self, hidden_size=5120, num_heads=24, num_kv_heads=4, head_dim=256,
                 intermediate_size=None, vocab_size=248320, eps=1e-6,
                 max_position=32768, rope_theta=10000.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size or 17408
        self.eps = eps
        # fc: [hidden; embed] -> hidden
        self.fc = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.pre_fc_norm_embedding = nn.Parameter(torch.ones(hidden_size))
        self.pre_fc_norm_hidden = nn.Parameter(torch.ones(hidden_size))
        # one decoder layer (full attention), Qwen3.5-style gated attention
        self.input_layernorm = nn.Parameter(torch.ones(hidden_size))
        self.post_attention_layernorm = nn.Parameter(torch.ones(hidden_size))
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim * 2, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = nn.Parameter(torch.ones(head_dim))
        self.k_norm = nn.Parameter(torch.ones(head_dim))
        self.gate_proj = nn.Linear(hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, hidden_size, bias=False)
        self.norm = nn.Parameter(torch.ones(hidden_size))
        # rotary (base cos/sin computed on the fly)
        self.rope_theta = rope_theta
        self.max_position = max_position
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # plain-RMSNorm convention by default (QAT ecosystem); Gemma = weight-1
        self.norm_gemma = False

    def _rms(self, x, w):
        var = x.float().pow(2).mean(-1, keepdim=True)
        xhat = (x.float() / (var + self.eps).sqrt()).to(x.dtype)
        return xhat * (w + 1 if self.norm_gemma else w)

    def _rope(self, positions):
        # positions [M] int64 -> cos/sin [M, head_dim] (full-dim, rotate_half style)
        freqs = positions.float()[:, None] * self.inv_freq[None, :]  # [M, D/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [M, D]
        return emb.cos(), emb.sin()

    @staticmethod
    def _apply_rope(q, k, cos, sin):
        # rotate_half convention (matches adallm _apply_rotary_pos_emb_next)
        def rot_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat([-x2, x1], dim=-1)
        cos = cos[:, None, :].to(q.dtype)
        sin = sin[:, None, :].to(q.dtype)
        return q * cos + rot_half(q) * sin, k * cos + rot_half(k) * sin

    def forward(self, h: torch.Tensor, embed_tok: torch.Tensor, positions: torch.Tensor,
                kv_cache: dict | None = None, attn_impl: str = "sdpa"):
        """h: target hidden states [M, H]; embed_tok: embedding of next token [M, H];
        positions: [M] absolute positions. kv_cache: {'k': [B, KV, L, D], 'v': ..., 'len': int}
        (simple python-managed cache for the probe; engine integration will use runner cache).
        Returns logits-ready hidden [M, H]."""
        # HF Qwen3-MTP convention: fc input = cat([embed(y_t), h_t])
        x_in = torch.cat([self._rms(embed_tok, self.pre_fc_norm_embedding),
                          self._rms(h, self.pre_fc_norm_hidden)], dim=-1)
        x = self.fc(x_in)
        # layer
        residual = x
        x = self._rms(x, self.input_layernorm)
        M = x.size(0)
        q_out = self.q_proj(x).view(M, self.num_heads, self.head_dim * 2)
        q, gate = q_out.split(self.head_dim, dim=-1)
        gate = gate.reshape(M, self.num_heads * self.head_dim)
        q = self._rms(q, self.q_norm)
        k = self._rms(self.k_proj(x).view(M, self.num_kv_heads, self.head_dim), self.k_norm)
        v = self.v_proj(x).view(M, self.num_kv_heads, self.head_dim)
        cos, sin = self._rope(positions)
        q, k = self._apply_rope(q, k, cos, sin)
        # append to cache
        if kv_cache is not None:
            L = kv_cache["len"]
            kv_cache["k"][:, :, L:L + M, :] = k.unsqueeze(1) if kv_cache["k"].size(1) == 1 else k.transpose(0, 1).unsqueeze(2).squeeze(3) if False else k.transpose(0,1)
            kv_cache["v"][:, :, L:L + M, :] = v.transpose(0, 1)
            k_full = kv_cache["k"][:, :, :L + M, :]
            v_full = kv_cache["v"][:, :, :L + M, :]
            kv_cache["len"] = L + M
        else:
            k_full = k.transpose(0, 1)
            v_full = v.transpose(0, 1)
        # sdpa: [B=1, H, L, D]
        o = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k_full.unsqueeze(2).expand(-1, -1, self.num_heads // self.num_kv_heads, -1, -1).reshape(1, self.num_heads, -1, self.head_dim) if k_full.size(1) == self.num_kv_heads else k_full.unsqueeze(0),
            v_full.unsqueeze(2).expand(-1, -1, self.num_heads // self.num_kv_heads, -1, -1).reshape(1, self.num_heads, -1, self.head_dim) if v_full.size(1) == self.num_kv_heads else v_full.unsqueeze(0),
        )
        o = o.squeeze(0).transpose(0, 1).reshape(M, self.num_heads * self.head_dim)
        o = o * torch.sigmoid(gate)
        attn_out = self.o_proj(o)
        x = residual + attn_out
        residual = x
        x = self._rms(x, self.post_attention_layernorm)
        g = self.gate_proj(x)
        u = self.up_proj(x)
        mlp = self.down_proj(F.silu(g) * u)
        x = residual + mlp
        return self._rms(x, self.norm)


KEY_MAP = {
    "language_model.mtp.fc.weight": "fc.weight",
    "language_model.mtp.pre_fc_norm_embedding.weight": "pre_fc_norm_embedding",
    "language_model.mtp.pre_fc_norm_hidden.weight": "pre_fc_norm_hidden",
    "language_model.mtp.layers.0.input_layernorm.weight": "input_layernorm",
    "language_model.mtp.layers.0.post_attention_layernorm.weight": "post_attention_layernorm",
    "language_model.mtp.layers.0.self_attn.q_proj.weight": "q_proj.weight",
    "language_model.mtp.layers.0.self_attn.k_proj.weight": "k_proj.weight",
    "language_model.mtp.layers.0.self_attn.v_proj.weight": "v_proj.weight",
    "language_model.mtp.layers.0.self_attn.o_proj.weight": "o_proj.weight",
    "language_model.mtp.layers.0.self_attn.q_norm.weight": "q_norm",
    "language_model.mtp.layers.0.self_attn.k_norm.weight": "k_norm",
    "language_model.mtp.layers.0.mlp.gate_proj.weight": "gate_proj.weight",
    "language_model.mtp.layers.0.mlp.up_proj.weight": "up_proj.weight",
    "language_model.mtp.layers.0.mlp.down_proj.weight": "down_proj.weight",
    "language_model.mtp.norm.weight": "norm",
}


def load_mtp(draft: MTPDraft, model_dir: str, device: torch.device, dtype=torch.float16):
    from safetensors.torch import load_file
    sd = load_file(f"{model_dir}/mtp.safetensors")
    mapped = {}
    for k, v in sd.items():
        if k not in KEY_MAP:
            raise KeyError(f"unmapped mtp key: {k}")
        mapped[KEY_MAP[k]] = v.to(device=device, dtype=dtype)
    # norm convention detection: QAT-family plain norms have mean ~1.23 (x*w),
    # Gemma-style ~0.23 (x*(1+w)). Sniff final norm.
    m = mapped["norm"].float().mean().item()
    draft.norm_gemma = m < 0.5
    if draft.norm_gemma:
        for nk in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "input_layernorm",
                   "post_attention_layernorm", "q_norm", "k_norm", "norm"):
            mapped[nk] = mapped[nk] - 1.0
    draft.load_state_dict(mapped, strict=True)
    return draft
