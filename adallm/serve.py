import json
import faulthandler
import os
import signal
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from adallm import AdaLLM, SamplingParams
from adallm.hf_utils import resolve_model_path


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str | list[str]
    max_tokens: int = 16
    temperature: float = 1.0
    seed: int | None = None
    ignore_eos: bool = False
    n: int = 1
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = 16
    temperature: float = 1.0
    seed: int | None = None
    ignore_eos: bool = False
    n: int = 1
    stream: bool = False
    chat_template_kwargs: dict | None = None


def _build_chat_prompt(messages: list[ChatMessage], tokenizer=None, template_kwargs: dict | None = None) -> str:
    # Prefer the model's real chat template (correct <|im_start|> framing,
    # thinking control via chat_template_kwargs). Fallback: naive transcript.
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        hf_messages = [{"role": m.role, "content": m.content} for m in messages]
        try:
            return tokenizer.apply_chat_template(
                hf_messages,
                tokenize=False,
                add_generation_prompt=True,
                **(template_kwargs or {}),
            )
        except Exception:
            pass
    parts: list[str] = []
    for msg in messages:
        role = msg.role.strip().lower()
        if role not in ("system", "user", "assistant"):
            role = "user"
        parts.append(f"{role}: {msg.content}")
    parts.append("assistant:")
    return "\n".join(parts)


def _split_thinking(text: str) -> tuple[str, str]:
    """Split '<think>...</think>' reasoning from visible content. Handles
    unterminated thinking (cut by max_tokens) by treating the rest as reasoning."""
    # Drop trailing stop/special markers that leak into decoded text.
    for marker in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
        text = text.replace(marker, "")
    text = text.strip()
    if "</think>" in text:
        idx = text.index("</think>")
        reasoning = text[:idx].replace("<think>", "", 1).strip()
        content = text[idx + len("</think>"):].strip()
        return content, reasoning
    if text.lstrip().startswith("<think>"):
        return "", text.replace("<think>", "", 1).strip()
    return text, ""


def _prompt_token_count(llm: AdaLLM, prompt: str | list[str]) -> int:
    if isinstance(prompt, str):
        return len(llm.tokenizer.encode(prompt))
    return sum(len(llm.tokenizer.encode(p)) for p in prompt)


def _completion_response(
    model_name: str,
    prompt_tokens: int,
    outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    completion_tokens = sum(len(out["token_ids"]) for out in outputs)
    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": idx,
                "text": out["text"],
                "finish_reason": "length",
            }
            for idx, out in enumerate(outputs)
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _chat_completion_response(
    model_name: str,
    prompt_tokens: int,
    outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    completion_tokens = sum(len(out["token_ids"]) for out in outputs)
    choices = []
    for idx, out in enumerate(outputs):
        content, reasoning = _split_thinking(out["text"])
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning:
            msg["reasoning_content"] = reasoning
        choices.append({
            "index": idx,
            "message": msg,
            "finish_reason": out.get("finish_reason") or "length",
        })
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _build_app(llm: AdaLLM, model_name: str) -> FastAPI:
    app = FastAPI()
    import queue as _queue
    import threading
    # Continuous-batching runner: a single engine thread steps the scheduler;
    # each HTTP request gets its own token queue keyed by seq_id.
    _req_lock = threading.Lock()
    _engine_lock = threading.Lock()    # serialized default (safe)
    _wakeup = threading.Event()
    _queues: dict[int, _queue.Queue] = {}

    def _engine_loop():
        import sys as _sys
        import traceback as _tb
        step_no = 0
        while True:
            _wakeup.wait()
            _wakeup.clear()
            while True:
                with _req_lock:
                    if llm.is_finished():
                        break
                step_no += 1
                try:
                    if os.getenv("ADALLM_DEBUG_ENGINE"):
                        print(f"[engine] step {step_no}", file=_sys.stderr, flush=True)
                    _, _, updates = llm.step(return_updates=True)
                except Exception:
                    # ANY engine failure (bad prompt, CUDA error, ...):
                    # surface to all waiting requests, drop their state.
                    _tb.print_exc()
                    with _req_lock:
                        for q in _queues.values():
                            q.put(("error", None))
                        # drop queued seqs from the scheduler
                        llm.scheduler.waiting.clear()
                        llm.scheduler.chunked.clear()
                        llm.scheduler.running.clear()
                        _queues.clear()
                    break
                for seq_id, token_id, finished in updates:
                    q = _queues.get(seq_id)
                    if q is not None:
                        if token_id != -1:
                            q.put(("token", token_id))
                        if finished:
                            q.put(("done", None))
                # NOTE: intermediate chunked-prefill steps return no updates
                # but still do work; the ONLY correct exit is is_finished().
    _use_queue_server = bool(os.getenv("ADALLM_QUEUE_SERVER", ""))
    if _use_queue_server:
        _engine_thread = threading.Thread(target=_engine_loop, daemon=True)
        _engine_thread.start()

    def _run_generate(prompts, sp):
        """Serialized generation (default): engine is single-threaded; the
        continuous-batching queue server (ADALLM_QUEUE_SERVER=1) is EXPERIMENTAL
        (known wedge under concurrent load, root-cause pending)."""
        if _use_queue_server:
            seq_id, q = _submit(prompts[0], sp)
            token_ids = []
            while True:
                kind, token_id = q.get()
                if kind in ("error", "done"):
                    if kind == "error":
                        raise RuntimeError("generation failed")
                    break
                token_ids.append(token_id)
            with _req_lock:
                _queues.pop(seq_id, None)
            return [{"text": llm.tokenizer.decode(token_ids), "token_ids": token_ids, "finish_reason": "stop"}]
        with _engine_lock:
            return llm.generate(prompts, sp, use_tqdm=False)

    def _submit(prompt: str, sp) -> tuple[int, _queue.Queue]:
        with _req_lock:
            if isinstance(prompt, str):
                ids = llm.tokenizer.encode(prompt)
            else:
                ids = prompt
            from adallm.runtime.sequence import Sequence as _Seq
            seq = _Seq(ids, sp)
            llm.scheduler.add(seq)
            q: _queue.Queue = _queue.Queue()
            _queues[seq.seq_id] = q
        _wakeup.set()
        return seq.seq_id, q

    @app.get("/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {"id": model_name, "object": "model"},
            ],
        }

    @app.post("/v1/completions")
    def completions(req: CompletionRequest) -> dict[str, Any]:
        if req.n != 1:
            raise HTTPException(status_code=400, detail="Only n=1 is supported.")
        prompts = req.prompt if isinstance(req.prompt, list) else [req.prompt]
        sp = SamplingParams(
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            ignore_eos=req.ignore_eos,
            seed=req.seed,
        )
        if req.stream:
            if len(prompts) != 1:
                raise HTTPException(status_code=400, detail="Streaming supports a single prompt.")
            created = int(time.time())
            stream_id = f"cmpl-{uuid.uuid4().hex}"

            def _stream():
                seq_id, q = _submit(prompts[0], sp)
                while True:
                    kind, token_id = q.get()
                    if kind in ("error", "done"):
                        break
                    text = llm.tokenizer.decode([token_id], skip_special_tokens=True)
                    if not text:
                        continue
                    chunk = {
                        "id": stream_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "text": text,
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(_stream(), media_type="text/event-stream")

        outputs = _run_generate(prompts, sp)
        prompt_tokens = _prompt_token_count(llm, prompts)
        return _completion_response(model_name, prompt_tokens, outputs)

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest) -> dict[str, Any]:
        if req.n != 1:
            raise HTTPException(status_code=400, detail="Only n=1 is supported.")
        prompt = _build_chat_prompt(req.messages, getattr(llm, "tokenizer", None), req.chat_template_kwargs)
        sp = SamplingParams(
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            ignore_eos=req.ignore_eos,
            seed=req.seed,
        )
        if req.stream:
            created = int(time.time())
            stream_id = f"chatcmpl-{uuid.uuid4().hex}"

            def _stream():
                sent_role = False
                in_thinking = False
                started_thinking = False
                pending = ""    # small buffer to detect <think>/</think> markers
                if not _use_queue_server:
                    _sent_role = False
                    for _, _, text, _ in llm.generate_stream([prompt], sp):
                        if not text:
                            continue
                        if not _sent_role:
                            _c = {"id": stream_id, "object": "chat.completion", "created": created,
                                  "model": model_name,
                                  "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
                            yield f"data: {json.dumps(_c)}\n\n"
                            _sent_role = True
                        _c = {"id": stream_id, "object": "chat.completion", "created": created,
                              "model": model_name,
                              "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
                        yield f"data: {json.dumps(_c)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                seq_id, q = _submit(prompt, sp)
                while True:
                    kind, token_id = q.get()
                    if kind == "error":
                        break
                    if kind == "done":
                        break
                    text = llm.tokenizer.decode([token_id], skip_special_tokens=True)
                    if not text:
                        continue
                    pending += text
                    # We may straddle a marker boundary; hold back a tail that
                    # could be the start of '<think>' or '</think>'.
                    holdback = min(len(pending), 8)
                    emit, tail = pending[:-holdback] if holdback else pending, pending[-holdback:] if holdback else ""
                    if not sent_role:
                        chunk = {
                            "id": stream_id,
                            "object": "chat.completion",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant"},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        sent_role = True
                    while True:
                        if not in_thinking:
                            idx = emit.find("<think>")
                            if idx == -1:
                                if emit:
                                    out = {"content": emit}
                                    chunk = {
                                        "id": stream_id,
                                        "object": "chat.completion",
                                        "created": created,
                                        "model": model_name,
                                        "choices": [{"index": 0, "delta": out, "finish_reason": None}],
                                    }
                                    yield f"data: {json.dumps(chunk)}\n\n"
                                break
                            if idx > 0:
                                out = {"content": emit[:idx]}
                                chunk = {
                                    "id": stream_id,
                                    "object": "chat.completion",
                                    "created": created,
                                    "model": model_name,
                                    "choices": [{"index": 0, "delta": out, "finish_reason": None}],
                                    }
                                yield f"data: {json.dumps(chunk)}\n\n"
                            emit = emit[idx + 7:]
                            in_thinking = True
                            started_thinking = True
                        else:
                            idx = emit.find("</think>")
                            if idx == -1:
                                if emit:
                                    out = {"reasoning_content": emit}
                                    chunk = {
                                        "id": stream_id,
                                        "object": "chat.completion",
                                        "created": created,
                                        "model": model_name,
                                        "choices": [{"index": 0, "delta": out, "finish_reason": None}],
                                    }
                                    yield f"data: {json.dumps(chunk)}\n\n"
                                break
                            emit = emit[idx + 8:]
                            in_thinking = False
                    pending = tail
                # flush remainder
                if pending:
                    out = {"reasoning_content": pending} if in_thinking else {"content": pending}
                    chunk = {
                        "id": stream_id,
                        "object": "chat.completion",
                        "created": created,
                        "model": model_name,
                        "choices": [{"index": 0, "delta": out, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(_stream(), media_type="text/event-stream")

        outputs = _run_generate([prompt], sp)
        prompt_tokens = _prompt_token_count(llm, prompt)
        return _chat_completion_response(model_name, prompt_tokens, outputs)

    return app


def serve(
    model: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    download_dir: str | None = None,
    max_model_len: int = 4096,
    max_num_seqs: int = 512,
    max_num_batched_tokens: int = 16384,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    enforce_eager: bool = False,
    kv_cache_dtype: str = "fp8",
    kv_cache_max_blocks: int = -1,
    fp8_decode_kernel_enabled: bool = True,
    fp8_decode_kernel_min_ctx: int = 0,
) -> None:
    model_path = resolve_model_path(model, download_dir)
    llm = AdaLLM(
        model_path,
        enforce_eager=enforce_eager,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        kv_cache_dtype=kv_cache_dtype,
        kv_cache_max_blocks=kv_cache_max_blocks,
        fp8_decode_kernel_enabled=fp8_decode_kernel_enabled,
        fp8_decode_kernel_min_ctx=fp8_decode_kernel_min_ctx,
    )
    app = _build_app(llm, model)
    # SIGUSR1 dumps all thread tracebacks (debug wedges without py-spy)
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    uvicorn.run(app, host=host, port=port)
