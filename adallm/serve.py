import json
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


def _build_chat_prompt(messages: list[ChatMessage]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg.role.strip().lower()
        if role not in ("system", "user", "assistant"):
            role = "user"
        parts.append(f"{role}: {msg.content}")
    parts.append("assistant:")
    return "\n".join(parts)


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
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": idx,
                "message": {
                    "role": "assistant",
                    "content": out["text"],
                },
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


def _build_app(llm: AdaLLM, model_name: str) -> FastAPI:
    app = FastAPI()

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
                for _, _, text, _ in llm.generate_stream(prompts, sp):
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

        outputs = llm.generate(prompts, sp, use_tqdm=False)
        prompt_tokens = _prompt_token_count(llm, prompts)
        return _completion_response(model_name, prompt_tokens, outputs)

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest) -> dict[str, Any]:
        if req.n != 1:
            raise HTTPException(status_code=400, detail="Only n=1 is supported.")
        prompt = _build_chat_prompt(req.messages)
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
                for _, _, text, _ in llm.generate_stream([prompt], sp):
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
                    if text:
                        chunk = {
                            "id": stream_id,
                            "object": "chat.completion",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(_stream(), media_type="text/event-stream")

        outputs = llm.generate([prompt], sp, use_tqdm=False)
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
        fp8_decode_kernel_enabled=fp8_decode_kernel_enabled,
        fp8_decode_kernel_min_ctx=fp8_decode_kernel_min_ctx,
    )
    app = _build_app(llm, model)
    uvicorn.run(app, host=host, port=port)
