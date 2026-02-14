import atexit
import os
import sys
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch
import torch.multiprocessing as mp

from adallm.settings import Config
from adallm.sampling import SamplingParams
from adallm.runtime.sequence import Sequence
from adallm.runtime.scheduler import Scheduler
from adallm.runtime.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        quiet = os.getenv("ADALLM_QUIET", "").strip().lower() in ("1", "true", "yes", "on")
        if not quiet:
            hf_cfg = config.hf_config
            arch = getattr(hf_cfg, "architectures", None)
            model_type = getattr(hf_cfg, "model_type", None)
            dtype = getattr(hf_cfg, "torch_dtype", None)
            kv_dtype = getattr(config, "kv_cache_dtype", None)
            nvfp4_mode = getattr(config, "nvfp4_mode", None)
            fp8_kernel = getattr(config, "fp8_decode_kernel_enabled", None)
            fp8_min_ctx = getattr(config, "fp8_decode_kernel_min_ctx", None)
            print(f"Model: {config.model}", file=sys.stderr, flush=True)
            print(
                f"Model type: {model_type}, architectures: {arch}",
                file=sys.stderr,
                flush=True,
            )
            print(
                f"Max model len: {config.max_model_len}, tensor parallel: {config.tensor_parallel_size}",
                file=sys.stderr,
                flush=True,
            )
            print(
                f"NVFP4 mode: {nvfp4_mode}, KV cache: {kv_dtype}, fp8 decode kernel: {fp8_kernel} (min ctx {fp8_min_ctx})",
                file=sys.stderr,
                flush=True,
            )
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model,
                use_fast=True,
                fix_mistral_regex=True,
            )
        except TypeError:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=False)
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        if not quiet:
            tok_name = type(self.tokenizer).__name__
            vocab_size = getattr(self.tokenizer, "vocab_size", None)
            print(f"Tokenizer: {tok_name} (vocab size {vocab_size})", file=sys.stderr, flush=True)
        config.eos = self._collect_eos_ids(self.tokenizer)
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        if not hasattr(self, "model_runner"):
            return
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self, return_updates: bool = False):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        updates = self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        if return_updates:
            return outputs, num_tokens, updates
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        seed = None
        for sp in sampling_params:
            if getattr(sp, "seed", None) is None:
                continue
            if seed is None:
                seed = sp.seed
            elif sp.seed != seed:
                raise ValueError("All sampling params must use the same seed.")
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs

    def generate_stream(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
    ):
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        seed = None
        for sp in sampling_params:
            if getattr(sp, "seed", None) is None:
                continue
            if seed is None:
                seed = sp.seed
            elif sp.seed != seed:
                raise ValueError("All sampling params must use the same seed.")
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        while not self.is_finished():
            _, _, updates = self.step(return_updates=True)
            for seq_id, token_id, finished in updates:
                text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                yield seq_id, token_id, text, finished

    @staticmethod
    def _collect_eos_ids(tokenizer) -> list[int]:
        eos_ids: list[int] = []
        raw_eos = getattr(tokenizer, "eos_token_id", None)
        if isinstance(raw_eos, (list, tuple, set)):
            eos_ids.extend(int(x) for x in raw_eos if x is not None)
        elif raw_eos is not None:
            eos_ids.append(int(raw_eos))
        for token in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>"):
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
            except Exception:
                token_id = None
            if token_id is None:
                continue
            if isinstance(token_id, list):
                eos_ids.extend(int(x) for x in token_id if x is not None)
            else:
                eos_ids.append(int(token_id))
        # drop unknown token ids
        try:
            unk_id = tokenizer.unk_token_id
        except Exception:
            unk_id = None
        eos_ids = [x for x in eos_ids if x is not None and x != unk_id]
        # unique while preserving order
        seen = set()
        ordered = []
        for x in eos_ids:
            if x in seen:
                continue
            seen.add(x)
            ordered.append(x)
        return ordered
