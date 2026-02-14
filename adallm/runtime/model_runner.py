import pickle
import torch
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

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        set_quant_config(hf_config, config)
        qcfg = get_quant_config()
        if not qcfg.enabled:
            raise RuntimeError("NVFP4-only runtime: quantization config is not NVFP4.")
        if qcfg.kv_cache_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise RuntimeError("NVFP4-only runtime: KV cache must be FP8.")
        if not qcfg.fp8_decode_kernel_enabled:
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
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * kv_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        if config.kv_cache_max_blocks > 0:
            config.num_kvcache_blocks = min(config.num_kvcache_blocks, config.kv_cache_max_blocks)
        assert config.num_kvcache_blocks > 0
        # Layout: [block, token, kv_head, head_dim] for both FP8 and FP16/BF16 caches.
        self.kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
            dtype=kv_dtype,
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                module.layer_id = layer_id
                layer_id += 1

    def fp8_decode_selftest(self):
        if self.enforce_eager:
            # Still compile once to fail fast on kernel issues.
            pass
        from adallm.ops.attention import fp8_kv_decode_group_kernel
        qcfg = get_quant_config()
        if not qcfg.fp8_decode_kernel_enabled:
            raise RuntimeError("FP8 decode kernel is disabled; this runtime requires it.")
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

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
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
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        max_seqlen_k = 0
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            seq_len = len(seq)
            context_lens.append(seq_len)
            if seq_len > max_seqlen_k:
                max_seqlen_k = seq_len
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables, max_seqlen_k=max_seqlen_k)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        bs = input_ids.size(0)
        context = get_context()
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
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
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs], max_seqlen_k=config.max_model_len)
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
