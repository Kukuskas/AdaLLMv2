#### terminal input:

```bash
cat > /tmp/bench_fp4.py <<'PY'
import os, time, torch
from adallm import AdaLLM, SamplingParams
model_path = os.path.abspath(os.path.join(os.getcwd(), "..", "Qwen3-8B-NVFP4"))

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
sp = SamplingParams(temperature=0.6, max_tokens=512, ignore_eos=True, seed=7)

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
python /tmp/bench_fp4.py
```

#### terminal output: 
, in a small village nestled between emerald hills, there lived a curious young girl named Lila. She was known for her boundless imagination and insatiable love for stories. While other children played in the fields or chased each other through the woods, Lila was always found with her nose buried in a book, or sketching fantastical creatures in her notebook. Her favorite tales were those of the Enchanted Forest, a place whispered about by the village elders and believed to be hidden somewhere beyond the misty mountains.

One crisp autumn morning, as the golden leaves danced around her, Lila decided it was time to seek the Enchanted Forest herself. Armed with a small satchel filled with provisions, a weathered map she had found in her grandfather's attic, and a heart full of dreams, she set off on her journey. The villagers watched her go, some with admiration, others with quiet concern, but Lila was undeterred. She had always felt a special connection to the forest, as if it called to her from the depths of her soul.

Her journey led her through dense woodlands, across winding rivers, and over rocky trails. Each step brought her closer to the heart of the forest, and with every passing day, she began to notice strange occurrences. The trees seemed to whisper secrets in the wind, the flowers bloomed in colors she had never seen before, and the air was thick with a magic that made her skin tingle. She was not alone, though. A gentle breeze carried the scent of something familiar, like the memory of her grandmother's lavender perfume. It was as if the forest itself was guiding her, urging her forward.

Days turned into weeks, and Lila's journey was filled with both wonder and peril. She encountered creatures of myth, some friendly, others not so much. A wise old owl named Orla became her companion, offering cryptic advice that often left her more confused than enlightened. She also met a mischievous sprite who loved to play pranks, and a solemn guardian of the forest who tested her intentions with riddles. Each challenge was a test of her courage, wisdom, and compassion.

But the deeper she ventured, the more she realized that the Ench'ted Forest was not just a place of magic; it was a living entity, with its own needs and desires. The forest was in trouble, its magic fading, and it needed a guardian to protect it. Lila, with her heart full of love and her spirit
---
tokens= 512 time= 13.437703609466553 tok/s= 38.10174825104103
current/peak GB 7.549653053283691 7.551072120666504