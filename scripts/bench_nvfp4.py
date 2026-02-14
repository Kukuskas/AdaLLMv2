import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adallm import AdaLLM, SamplingParams

DEFAULT_PROMPTS = [
    "Once upon a time",
    "The quick brown fox",
    "In a galaxy far, far away",
    "Dear Sir or Madam, I am writing to",
    "The mitochondria is the",
    "To be or not to be, that is",
    "Breaking news: Scientists have discovered",
    "Step 1: Preheat the oven to",
    "The capital of France is",
    "def fibonacci(n):",
    "According to the laws of physics,",
    "Roses are red, violets are",
    "The year was 2045, and humanity had finally",
    "SELECT * FROM users WHERE",
    "In conclusion, the evidence suggests that",
    "{\"name\": \"John\", \"age\":",
]


def _load_prompts(prompt: str, use_prompt_set: bool, prompts_file: str | None) -> list[str]:
    if prompts_file:
        with open(prompts_file, "r", encoding="utf-8") as handle:
            prompts = [line.strip() for line in handle if line.strip()]
        if prompts:
            return prompts
    if use_prompt_set:
        return list(DEFAULT_PROMPTS)
    return [prompt]


def _make_batch_prompts(prompts: list[str], batch_size: int) -> list[str]:
    if len(prompts) >= batch_size:
        return prompts[:batch_size]
    batched = []
    for i in range(batch_size):
        base = prompts[i % len(prompts)]
        batched.append(f"{base} #{i}")
    return batched


def run_bench(
    model_path: str,
    batch_sizes: list[int],
    max_tokens: int,
    prompt: str,
    seed: int | None,
    use_prompt_set: bool,
    prompts_file: str | None,
):
    llm = AdaLLM(
        model_path,
        enforce_eager=False,
        max_model_len=4096,
        kv_cache_dtype="fp8",
        fp8_decode_kernel_enabled=True,
        fp8_decode_kernel_min_ctx=0,
    )
    llm.generate(["warmup"], SamplingParams(max_tokens=4, ignore_eos=True), use_tqdm=False)

    results = []
    for bs in batch_sizes:
        torch.cuda.reset_peak_memory_stats()
        prompts = _make_batch_prompts(_load_prompts(prompt, use_prompt_set, prompts_file), bs)
        sp = SamplingParams(temperature=0.6, max_tokens=max_tokens, ignore_eos=True, seed=seed)

        t0 = time.time()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        torch.cuda.synchronize()
        dt = time.time() - t0

        total_tokens = sum(len(out["token_ids"]) for out in outs)
        tok_s = total_tokens / dt
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        results.append((bs, total_tokens, dt, tok_s, peak_gb))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--use-prompt-set", action="store_true")
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    model_path = os.path.abspath(args.model_path)

    results = run_bench(
        model_path,
        batch_sizes,
        args.max_tokens,
        args.prompt,
        args.seed,
        args.use_prompt_set,
        args.prompts_file,
    )

    print("batch,total_tokens,seconds,tok_per_s,peak_gb")
    for bs, total_tokens, dt, tok_s, peak_gb in results:
        print(f"{bs},{total_tokens},{dt:.4f},{tok_s:.2f},{peak_gb:.2f}")


if __name__ == "__main__":
    main()
