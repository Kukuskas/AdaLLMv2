import os
import pickle
import sys
import torch

# Pinned staging allocations can deadlock when created from a non-main engine
# thread (cudaHostAlloc cross-thread contention). Main thread may pin; the
# serve engine thread must not.
_PIN = os.getenv("ADALLM_NO_PIN", "") != "1"
# Round4-B (default OFF): capture lm_head + sampler into the decode CUDA
# graphs (ADALLM_GRAPH_SAMPLE=1). Today only self.model() is captured;
# compute_logits (marlin lm_head) + the ~8-kernel eager sampler run outside
# the graph every decode step.
_GRAPH_SAMPLE_ENV = os.getenv("ADALLM_GRAPH_SAMPLE", "0").strip().lower()
GRAPH_SAMPLE_ENABLED = _GRAPH_SAMPLE_ENV not in ("0", "false", "off", "")
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from adallm.settings import Config
from adallm.runtime.sequence import Sequence
from adallm.architectures.registry import build_model
from adallm.ops.sampler import Sampler
from adallm.infra.context import set_context, get_context, reset_context
from adallm.infra.loader import load_model
from adallm.infra.quant import set_quant_config, get_quant_config


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        hf_config.max_model_len = config.max_model_len
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:" + os.getenv("ADALLM_DIST_PORT", "2333"), world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        # Round2-D: fp16 activations end-to-end (ADALLM_FP16_ACT=1). The
        # Marlin / fp8-KV / Triton kernels all consume fp16; the bf16 default
        # forced a bf16<->fp16 convert pair on nearly every GEMM boundary
        # (~1380 tiny copy kernels/step, ~2.6 ms). Building the model with a
        # fp16 default dtype removes them. Delta vs bf16: fp16 has less
        # exponent range — activation-overflow risk exists in principle, but
        # Gemma-style norms keep residual magnitudes ~O(10), and quality
        # checks ("17*23"->391, logits compare) passed.
        if os.getenv("ADALLM_FP16_ACT", "0").strip().lower() not in ("0", "false", "off"):
            hf_config.torch_dtype = torch.float16
            torch.set_default_dtype(torch.float16)
        torch.set_default_device("cuda")
        set_quant_config(hf_config, config)
        qcfg = get_quant_config()
        if not qcfg.enabled:
            raise RuntimeError("NVFP4-only runtime: quantization config is not NVFP4.")
        if qcfg.kv_cache_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2, torch.float16, torch.bfloat16):
            raise RuntimeError("NVFP4-only runtime: KV cache must be FP8 or fp16/bf16.")
        if qcfg.kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and not qcfg.fp8_decode_kernel_enabled:
            raise RuntimeError("NVFP4-only runtime: fp8_decode_kernel_enabled must be True.")
        hf_config.moe_offload_experts = bool(getattr(config, "moe_offload_experts", False))
        hf_config.moe_gpu_expert_capacity = int(getattr(config, "moe_gpu_expert_capacity", 0))
        hf_config.moe_offload_pin_memory = bool(getattr(config, "moe_offload_pin_memory", False))
        hf_config.moe_offload_prefetch = bool(getattr(config, "moe_offload_prefetch", False))
        hf_config.moe_offload_prefetch_window = int(getattr(config, "moe_offload_prefetch_window", 0))
        hf_config.moe_offload_pack = bool(getattr(config, "moe_offload_pack", True))
        hf_config.moe_offload_pack_capacity = int(getattr(config, "moe_offload_pack_capacity", 0))
        hf_config.moe_offload_pack_cache = bool(getattr(config, "moe_offload_pack_cache", True))
        hf_config.moe_offload_pack_reuse = bool(getattr(config, "moe_offload_pack_reuse", True))
        hf_config.moe_offload_pack_gather_min_rows = int(getattr(config, "moe_offload_pack_gather_min_rows", 128))
        hf_config.moe_cache_decay = float(getattr(config, "moe_cache_decay", 0.95))
        hf_config.moe_cache_decay_interval = int(getattr(config, "moe_cache_decay_interval", 128))
        hf_config.moe_profile = bool(getattr(config, "moe_profile", False))
        hf_config.moe_profile_interval = int(getattr(config, "moe_profile_interval", 1))
        hf_config.moe_profile_layers = getattr(config, "moe_profile_layers", None)
        self.model, self.is_moe = build_model(hf_config)
        if self.is_moe and hf_config.moe_offload_experts:
            self.enforce_eager = True
        load_model(self.model, config.model)
        self.sampler = Sampler()
        # Materialize lazily-created device constants (FP4 LUT) before any
        # CUDA graph capture; also runs them once for Triton JIT warmup.
        from adallm.ops.nvfp4_gemm import warm_cudagraph_buffers
        warm_cudagraph_buffers(torch.device("cuda", torch.cuda.current_device()))
        self.warmup_model()
        self.allocate_kv_cache()
        self.fp8_decode_selftest()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="adallm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="adallm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if self.config.warmup_tokens <= 0:
            return
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        warmup_len = min(self.config.warmup_tokens, max_model_len)
        num_seqs = min(
            self.config.warmup_num_seqs,
            max_num_batched_tokens // warmup_len,
            self.config.max_num_seqs,
        )
        if num_seqs <= 0:
            return
        seqs = [Sequence([0] * warmup_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        qcfg = get_quant_config()
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        kv_dtype = qcfg.kv_cache_dtype if qcfg.enabled else hf_config.torch_dtype
        if os.getenv("ADALLM_DEBUG_KV", ""):
            print(f"[kv] free={free/2**30:.2f}GiB total={total/2**30:.2f}GiB used={used/2**30:.2f}GiB "
                  f"peak={peak/2**30:.2f}GiB current={current/2**30:.2f}GiB util={config.gpu_memory_utilization}",
                  file=sys.stderr, flush=True)
        # Hybrid models: only full-attention layers hold a KV cache.
        layer_types = getattr(hf_config, "layer_types", None)
        if layer_types:
            num_kv_layers = sum(1 for t in layer_types if t == "full_attention")
        else:
            num_kv_layers = hf_config.num_hidden_layers
        block_bytes = 2 * num_kv_layers * self.block_size * num_kv_heads * head_dim * kv_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        if config.kv_cache_max_blocks > 0:
            config.num_kvcache_blocks = min(config.num_kvcache_blocks, config.kv_cache_max_blocks)
        if os.getenv("ADALLM_DEBUG_KV", ""):
            print(f"[kv] num_kv_layers={num_kv_layers} block_bytes={block_bytes/2**20:.2f}MiB "
                  f"num_kvcache_blocks={config.num_kvcache_blocks} cap={config.kv_cache_max_blocks}",
                  file=sys.stderr, flush=True)
        assert config.num_kvcache_blocks > 0
        # Layout: [block, token, kv_head, head_dim] for both FP8 and FP16/BF16 caches.
        self.kv_cache = torch.empty(
            2,
            num_kv_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
            dtype=kv_dtype,
        )
        # Delta-net (mamba-style) state pools for hybrid models. One slot per
        # concurrent sequence (+ reserved NULL slot 0) keeps state updates
        # inside kernels → CUDA-graph safe and multi-sequence correct.
        from adallm.architectures.qwen3_5 import Qwen3_5GatedDeltaNet
        delta_modules = [m for m in self.model.modules() if isinstance(m, Qwen3_5GatedDeltaNet)]
        num_delta_slots = config.max_num_seqs + 2
        # ensure pool covers every CUDA-graph batch size (padded rows may use
        # slot ids up to graph_bs; NULL rows use 0)
        num_delta_slots = max(num_delta_slots, (min(config.max_num_seqs, 512) + 2))
        state_dtype = hf_config.torch_dtype if hf_config is not None else torch.bfloat16
        for m in delta_modules:
            m.allocate_state_pool(num_delta_slots, self.kv_cache.device, state_dtype)
        self.delta_num_layers = len(delta_modules)
        self.delta_modules = delta_modules
        self.graph_delta_slots = torch.zeros(num_delta_slots, dtype=torch.int64, device=self.kv_cache.device)
        layer_id = 0
        import os as _os
        self.kv_shadow_enabled = bool(_os.environ.get("ADALLM_KV_SHADOW"))
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                module.layer_id = layer_id
                if self.kv_shadow_enabled and kv_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    module.k_shadow = torch.zeros_like(module.k_cache, dtype=torch.float16)
                    module.v_shadow = torch.zeros_like(module.v_cache, dtype=torch.float16)
                layer_id += 1

    def fp8_decode_selftest(self):
        if self.enforce_eager:
            # Still compile once to fail fast on kernel issues.
            pass
        qcfg = get_quant_config()
        if qcfg.kv_cache_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            return  # fp16/bf16 KV cache: decode uses flash paged path, nothing to selftest
        if not qcfg.fp8_decode_kernel_enabled:
            raise RuntimeError("FP8 decode kernel is disabled; this runtime requires it.")
        from adallm.ops.attention import fp8_kv_decode_group_kernel
        # Use the first attention module's cache to compile the kernel.
        attn = next(m for m in self.model.modules() if hasattr(m, "k_cache") and hasattr(m, "v_cache"))
        k_cache = attn.k_cache
        v_cache = attn.v_cache
        if k_cache.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise RuntimeError("FP8 decode selftest requires FP8 KV cache.")
        block_size = k_cache.shape[1]
        num_kv_heads = k_cache.shape[2]
        head_dim = k_cache.shape[3]
        num_heads = attn.num_heads
        q_per_kv = num_heads // num_kv_heads
        block_q = 1 << (q_per_kv - 1).bit_length()
        bs = 1
        q = torch.zeros((bs, num_heads, head_dim), device=k_cache.device, dtype=torch.float16)
        out = torch.empty_like(q)
        block_tables = torch.zeros((bs, 1), device=k_cache.device, dtype=torch.int32)
        context_lens = torch.ones((bs,), device=k_cache.device, dtype=torch.int32)
        try:
            fp8_kv_decode_group_kernel[(bs * num_kv_heads,)](
                q,
                k_cache,
                v_cache,
                out,
                block_tables,
                context_lens,
                attn.scale,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                k_cache.stride(0),
                k_cache.stride(1),
                k_cache.stride(2),
                k_cache.stride(3),
                v_cache.stride(0),
                v_cache.stride(1),
                v_cache.stride(2),
                v_cache.stride(3),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                block_tables.stride(0),
                bs,
                MAX_BLOCKS=block_tables.shape[1],
                BLOCK_SIZE=block_size,
                HEAD_DIM=head_dim,
                NUM_HEADS=num_heads,
                NUM_KV_HEADS=num_kv_heads,
                Q_PER_KV=q_per_kv,
                BLOCK_Q=block_q,
                WINDOW_SIZE=0,
            )
            torch.cuda.synchronize()
        except Exception as exc:
            raise RuntimeError("FP8 KV decode kernel selftest failed.") from exc

    def _reset_delta_slot(self, slot: int):
        """Zero a recycled delta-net state slot (all layers) for a fresh sequence."""
        for m in self.delta_modules:
            if m.mamba_state is not None:
                m.mamba_state[slot].zero_()
                m.conv_state[slot].zero_()

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=_PIN).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        delta_slots = []
        for seq in seqs:
            start = seq.num_cached_tokens + seq.num_prefilled_tokens
            chunk = seq.cur_chunk if seq.cur_chunk > 0 else len(seq) - start
            input_ids.extend(seq[start : start + chunk])
            positions.extend(list(range(start, start + chunk)))
            seqlen_q = chunk
            seqlen_k = start + chunk
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if seq.delta_slot > 0 and seq.num_prefilled_tokens == 0 and seq.num_cached_tokens == 0:
                self._reset_delta_slot(seq.delta_slot)    # fresh seq: zero recycled state
            delta_slots.append(seq.delta_slot)
            if not seq.block_table:    # warmup
                continue
            # slot_mapping must cover ONLY this chunk's tokens (chunked prefill)
            tok_begin, tok_end = start, start + chunk
            blk_begin = tok_begin // self.block_size
            blk_end = (tok_end - 1) // self.block_size
            for bi in range(blk_begin, blk_end + 1):
                blk_tok_lo = bi * self.block_size
                blk_tok_hi = min(blk_tok_lo + self.block_size, seq.num_tokens)
                lo = max(blk_tok_lo, tok_begin)
                hi = min(blk_tok_hi, tok_end)
                block_id = seq.block_table[bi]
                base = block_id * self.block_size
                slot_mapping.extend(list(range(base + (lo - blk_tok_lo), base + (hi - blk_tok_lo))))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=_PIN).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=_PIN).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=_PIN).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=_PIN).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=_PIN).cuda(non_blocking=True)
        delta_slots = torch.tensor(delta_slots, dtype=torch.int64, pin_memory=_PIN).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables, delta_slots)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        delta_slots = []
        max_seqlen_k = 0
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            seq_len = len(seq)
            context_lens.append(seq_len)
            delta_slots.append(seq.delta_slot)
            if seq_len > max_seqlen_k:
                max_seqlen_k = seq_len
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=_PIN).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=_PIN).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=_PIN).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=_PIN).cuda(non_blocking=True)
        delta_slots = torch.tensor(delta_slots, dtype=torch.int64, pin_memory=_PIN).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables, max_seqlen_k=max_seqlen_k, delta_slots=delta_slots)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=_PIN).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def capture_verify_graph(self, span_rows: int = 4):
        """Round8-A (lever 1): CUDA graph for the MTP verify forward — a
        FIXED-shape prefill-continuation of `span_rows` tokens (k+1) for a
        single sequence. Static buffers: input_ids/positions (copied in per
        step), slot_mapping, cu_seqlens (fixed [0, span] / [0, ctx]),
        block_tables row, delta slot; output hidden states exposed via
        self.verify_graph_vars so the spec engine reads them without an
        eager forward. max_seqlen_k is static (full context) — flash varlen
        only uses it as a hint, correctness unaffected."""
        if self.enforce_eager:
            return False
        dev = "cuda"
        bs = span_rows
        max_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size
        gv = {
            "input_ids": torch.zeros(bs, dtype=torch.int64, device=dev),
            "positions": torch.arange(bs, dtype=torch.int64, device=dev),
            "slot_mapping": torch.zeros(bs, dtype=torch.int32, device=dev),
            "cu_q": torch.tensor([0, bs], dtype=torch.int32, device=dev),
            "cu_k": torch.tensor([0, self.config.max_model_len], dtype=torch.int32, device=dev),
            "block_tables": torch.zeros(1, max_blocks, dtype=torch.int32, device=dev),
            "n_blocks_row": torch.zeros(1, dtype=torch.int32, device=dev),
            "delta_slot": torch.zeros(1, dtype=torch.int64, device=dev),
            "hidden": torch.zeros(bs, self.config.hf_config.hidden_size, device=dev),
        }
        # cu_seqlens_k must reflect actual k length: block-table gather path
        # (prefix cache) uses cu_k[1:] - cu_k[:-1] as block counts. Use a
        # dedicated static "k_len" we update per replay via n_blocks_row.
        # For the no-prefix-cache path flash reads cu_k directly: we cap it
        # with a large but valid value (max_model_len) — kernel treats it as
        # max hint only when block_tables present.
        try:
            # warmup
            set_context(True,
                        cu_seqlens_q=gv["cu_q"], cu_seqlens_k=gv["cu_k"],
                        max_seqlen_q=bs, max_seqlen_k=self.config.max_model_len,
                        slot_mapping=gv["slot_mapping"], context_lens=None,
                        block_tables=gv["block_tables"],
                        delta_slots=gv["delta_slot"])
            gv["hidden"] = self.model(gv["input_ids"], gv["positions"])
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, self.graph_pool):
                gv["hidden"] = self.model(gv["input_ids"], gv["positions"])
            self.verify_graph = graph
            self.verify_graph_vars = gv
            reset_context()
            return True
        except Exception as e:
            print(f"[verify-graph] capture FAILED ({e!r}); MTP stays eager", flush=True)
            reset_context()
            return False

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        bs = input_ids.size(0)
        context = get_context()
        graph_key = next((x for x in self.graph_bs if x >= bs), None)
        if graph_key is None:
            # bs exceeds the largest captured graph (e.g. max_num_seqs between
            # graph shapes like 9..15) — fall back to eager rather than crash.
            return self.model.compute_logits(self.model(input_ids, positions))
        graph, slots_buf = self.graphs[graph_key]
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        slots_buf[:bs] = context.delta_slots[:bs]
        slots_buf[bs:] = 0    # padding rows → NULL slot (kernels skip them)
        if GRAPH_SAMPLE_ENABLED:
            # Round4-B: the graph captured model+lm_head+sampler; copy the
            # sampled token ids out of the graph's static buffer.
            graph.replay()
            return graph_vars["sample_ids"][:bs]
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        if self.rank == 0:
            temperatures = self.prepare_sample(seqs)
            if GRAPH_SAMPLE_ENABLED and not is_prefill and not self.enforce_eager and len(seqs) <= 512:
                # copy into the graph's static buffer (fp32, length max graph bs)
                gv = self.graph_vars
                gv["temperatures"].fill_(1.0)
                gv["temperatures"][: len(seqs)] = temperatures
                out = self.run_model(input_ids, positions, is_prefill)
                token_ids = out.tolist()
            else:
                logits = self.run_model(input_ids, positions, is_prefill)
                token_ids = self.sampler(logits, temperatures).tolist()
        else:
            self.run_model(input_ids, positions, is_prefill)
            token_ids = None
        if is_prefill and token_ids is not None:
            # intermediate chunked-prefill chunks advance the cursor only
            token_ids = [tid if seq.chunk_final else -1 for seq, tid in zip(seqs, token_ids)]
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [bs for bs in ([1, 2, 4, 8] + list(range(16, max_bs + 1, 16))) if bs <= max_bs]
        self.graphs = {}
        self.graph_pool = None
        self.verify_graph = None
        self.verify_graph_vars = None
        # Round4-B: static buffers for in-graph lm_head + sampler
        sample_ids = torch.zeros(max_bs, dtype=torch.int64)
        temperatures = torch.ones(max_bs, dtype=torch.float32)

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            # Persistent per-batch delta slot buffer: run_model copies actual
            # slot ids here before replay; kernels read them on-device.
            slots_buf = torch.zeros(bs, dtype=torch.int64, device="cuda")
            self.graph_delta_slots[:bs] = 0    # all NULL during capture warmup
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs], max_seqlen_k=config.max_model_len, delta_slots=slots_buf)
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            if GRAPH_SAMPLE_ENABLED:
                logits = self.model.compute_logits(outputs[:bs])         # warmup
                self.sampler(logits, temperatures[:bs])
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
                if GRAPH_SAMPLE_ENABLED:
                    # lm_head + sampler inside the graph (Round4-B): removes
                    # ~15 eager kernel launches + their gaps per decode step.
                    logits = self.model.compute_logits(outputs[:bs])
                    sample_ids[:bs] = self.sampler(logits, temperatures[:bs])
            self.graphs[bs] = (graph, slots_buf)
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
            sample_ids=sample_ids,
            temperatures=temperatures,
        )
