"""Round4-B: FP8 row-wise quantized embedding table (ADALLM_EMBED_FP8=1,
default OFF).

The 248320x5120 fp16 embedding is 2.4 GiB and pure lookup bandwidth at
decode (bs rows of 5120). Row-wise e4m3 + fp32 per-row scale halves it to
1.2 GiB AND halves the gather bytes. The dequant convert happens on the
tiny [bs, 5120] result, not the table.

Quality gates: 17*23 -> 391, palindrome prompt coherent, greedy text compare
vs fp16 embedding (see probes/round4-b/arm_embedfp8*).
"""
import os
import torch

EMBED_FP8_ENABLED = os.getenv("ADALLM_EMBED_FP8", "0").strip().lower() not in ("0", "false", "off", "")
# keep the fp16 table resident for A/B (normally freed)
_KEEP_FP16 = os.getenv("ADALLM_EMBED_KEEP_FP16", "0") == "1"


@torch.no_grad()
def repack_embed(module) -> bool:
    """Quantize a VocabParallelEmbedding's weight row-wise to fp8 in place."""
    W = module.weight.data
    if W.dtype not in (torch.float16, torch.bfloat16):
        return False
    V, K = W.shape
    dev = W.device
    Wf = W.float()
    wmax = Wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = wmax / 448.0
    Wq = (Wf / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    module._fp8_embed = {
        "wq": Wq.contiguous(),
        "scale": scale.squeeze(1).contiguous(),   # fp32 [V]
        "orig_dtype": W.dtype,
    }
    if _KEEP_FP16:
        module._fp8_embed["w16"] = W
    else:
        module.weight.data = torch.empty(0, device=dev, dtype=W.dtype)
    import math
    gib = (V * K * Wq.element_size() + V * 4) / 2**30
    print(f"[embed-fp8] quantized embedding {V}x{K}: {gib:.2f} GiB (was {V*K*2/2**30:.2f} GiB), "
          f"rowmax range [{wmax.min().item():.4f}, {wmax.max().item():.4f}]", flush=True)
    return True


def embed_fp8_lookup(module, x: torch.Tensor) -> torch.Tensor:
    """Row-gather from the fp8 table, then dequantize the tiny result."""
    m = module._fp8_embed
    rows = m["wq"].index_select(0, x)              # fp8 [.., K]
    return rows.to(torch.float16) * m["scale"].index_select(0, x).unsqueeze(-1)
