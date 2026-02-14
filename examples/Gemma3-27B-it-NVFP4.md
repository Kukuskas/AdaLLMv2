#### terminal input:
```bash
cat > /tmp/bench_fp4_gemma.py <<'PY'
import os, time, torch
from adallm import AdaLLM, SamplingParams
model_path = os.path.abspath(os.path.join(os.getcwd(), "..", "Gemma3-27B-it-NVFP4"))

llm = AdaLLM(
    model_path,
    enforce_eager=False,
    max_model_len=4096,
    kv_cache_dtype="fp8",
    fp8_decode_kernel_enabled=True,
    fp8_decode_kernel_min_ctx=0,
)
llm.generate(["warmup"], SamplingParams(max_tokens=4, ignore_eos=True), use_tqdm=False)

torch.cuda.reset_peak_memory_stats()
prompt = "Once upon a time"
sp = SamplingParams(temperature=0.6, max_tokens=128, ignore_eos=True)

t0 = time.time()
outs = llm.generate([prompt], sp, use_tqdm=False)
torch.cuda.synchronize()
dt = time.time() - t0

print(outs[0]["text"])
print("---")
print("tokens=", len(outs[0]["token_ids"]), "time=", dt, "tok/s=", len(outs[0]["token_ids"])/dt)
print("current/peak GB", torch.cuda.memory_allocated()/1024**3, torch.cuda.max_memory_allocated()/1024**3)
PY

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 NVFP4_FP8=1 \
python /tmp/bench_fp4_gemma.py
```


#### terminal output:
, in a land filled with rolling hills and sparkling rivers, lived a young woman named Elara. She was known throughout the kingdom for her kindness and her gift for healing. But Elara was also burdened with a secret - she could speak to plants. 

Not in a booming, commanding voice, but in gentle whispers that only the leaves and blossoms understood. They told her of the earth's aches, of thirst and of sickness, and Elara, in turn, did everything she could to soothe them.

One day, a shadow fell upon the land. The King's beloved Rose Garden, the pride of the kingdom
---
tokens= 128 time= 9.38269591331482 tok/s= 13.642134540282548
current/peak GB 19.829191207885742 19.83246088027954
