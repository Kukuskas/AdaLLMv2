import argparse
import os
import sys
try:
    import readline  # noqa: F401
except Exception:
    readline = None

from adallm import AdaLLM, SamplingParams
from adallm.hf_utils import resolve_model_path
from adallm.serve import serve


_BANNER = r"""
   █████████       █████           █████       █████       ██████   ██████
  ███░░░░░███     ░░███           ░░███       ░░███       ░░██████ ██████ 
 ░███    ░███   ███████   ██████   ░███        ░███        ░███░█████░███ 
 ░███████████  ███░░███  ░░░░░███  ░███        ░███        ░███░░███ ░███ 
 ░███░░░░░███ ░███ ░███   ███████  ░███        ░███        ░███ ░░░  ░███ 
 ░███    ░███ ░███ ░███  ███░░███  ░███      █ ░███      █ ░███      ░███ 
 █████   █████░░████████░░████████ ███████████ ███████████ █████     █████
░░░░░   ░░░░░  ░░░░░░░░  ░░░░░░░░ ░░░░░░░░░░░ ░░░░░░░░░░░ ░░░░░     ░░░░░ 
                                                                          
"""


def _print_banner() -> None:
    quiet = os.getenv("ADALLM_QUIET", "").strip().lower() in ("1", "true", "yes", "on")
    if quiet:
        return
    print(_BANNER, file=sys.stderr)
    print("Version 0.1.0", file=sys.stderr, flush=True)
    print("Author: Ayoub Ben Chaliah", file=sys.stderr)
    print("Email: ayoub1benchaliah@gmail.com", file=sys.stderr, flush=True)


class _AdaLLMParser(argparse.ArgumentParser):
    def format_help(self) -> str:
        text = super().format_help()
        subparsers = None
        for action in self._actions:
            if isinstance(action, argparse._SubParsersAction):
                subparsers = action
                break
        if not subparsers:
            return text
        for _, subparser in subparsers.choices.items():
            text += "\n" + subparser.format_help()
        return text


def _add_serve_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "serve",
        help="Start an OpenAI-compatible server for AdaLLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model",
        help="Local path or Hugging Face repo ID.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--download-dir", default=None)

    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")

    parser.add_argument("--kv-cache-dtype", default="fp8")
    parser.add_argument("--kv-cache-max-blocks", type=int, default=-1)
    parser.add_argument(
        "--fp8-decode-kernel-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fp8-decode-kernel-min-ctx", type=int, default=0)

    parser.set_defaults(func=_handle_serve)


def _add_run_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "run",
        help="Run a prompt and stream tokens to stdout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model",
        help="Local path or Hugging Face repo ID.",
    )
    parser.add_argument("prompt", nargs="?", help="Prompt text. If omitted, read from stdin.")
    parser.add_argument("--download-dir", default=None)

    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")

    parser.add_argument("--kv-cache-dtype", default="fp8")
    parser.add_argument(
        "--fp8-decode-kernel-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fp8-decode-kernel-min-ctx", type=int, default=0)

    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--chat",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use chat templates and keep history when interactive.",
    )

    parser.set_defaults(func=_handle_run)


def _handle_serve(args: argparse.Namespace) -> None:
    serve(
        model=args.model,
        host=args.host,
        port=args.port,
        download_dir=args.download_dir,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        kv_cache_dtype=args.kv_cache_dtype,
        kv_cache_max_blocks=args.kv_cache_max_blocks,
        fp8_decode_kernel_enabled=args.fp8_decode_kernel_enabled,
        fp8_decode_kernel_min_ctx=args.fp8_decode_kernel_min_ctx,
    )


def _handle_run(args: argparse.Namespace) -> None:
    model_path = resolve_model_path(args.model, args.download_dir)
    llm = AdaLLM(
        model_path,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        kv_cache_dtype=args.kv_cache_dtype,
        kv_cache_max_blocks=args.kv_cache_max_blocks,
        fp8_decode_kernel_enabled=args.fp8_decode_kernel_enabled,
        fp8_decode_kernel_min_ctx=args.fp8_decode_kernel_min_ctx,
    )
    sp = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
        seed=args.seed,
    )

    def _format_chat_prompt(messages: list[dict[str, str]]) -> str:
        tokenizer = llm.tokenizer
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        parts = []
        for msg in messages:
            parts.append(f"{msg['role']}: {msg['content']}")
        parts.append("assistant:")
        return "\n".join(parts)

    def _run_prompt(text: str, messages: list[dict[str, str]] | None) -> None:
        if messages is not None:
            messages.append({"role": "user", "content": text})
            prompt = _format_chat_prompt(messages)
            prompt_input = llm.tokenizer.encode(prompt, add_special_tokens=False)
            if not prompt_input:
                raise SystemExit("Prompt is empty after chat templating.")
        else:
            prompt_input = text

        token_ids: list[int] = []
        active_seq_id = None
        for seq_id, token_id, chunk, _ in llm.generate_stream([prompt_input], sp):
            if active_seq_id is None:
                active_seq_id = seq_id
            if seq_id != active_seq_id:
                continue
            token_ids.append(token_id)
            if chunk:
                sys.stdout.write(chunk)
                sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()
        if messages is not None and token_ids:
            assistant_text = llm.tokenizer.decode(token_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": assistant_text})

    chat_mode = args.chat
    if chat_mode is None:
        chat_mode = args.prompt is None and sys.stdin.isatty()
    messages = [] if chat_mode else None

    if args.prompt is not None:
        if not args.prompt.strip():
            raise SystemExit("Prompt is empty.")
        _run_prompt(args.prompt, messages)
        return

    if not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
        if not prompt:
            raise SystemExit("Prompt is empty.")
        _run_prompt(prompt, messages)
        return

    print("Enter prompts. Type \\q to quit.", file=sys.stderr)
    if readline is not None:
        try:
            readline.set_history_length(1000)
        except Exception:
            pass
    while True:
        try:
            line = input(">>> ")
        except EOFError:
            break
        except KeyboardInterrupt:
            sys.stdout.write("\n")
            break
        if line.strip() == "\\q":
            break
        if not line.strip():
            continue
        _run_prompt(line, messages)


def main() -> None:
    _print_banner()
    parser = _AdaLLMParser(
        prog="adallm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_serve_parser(subparsers)
    _add_run_parser(subparsers)
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]) and len(sys.argv) == 2:
        print(parser.format_help())
        return
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
