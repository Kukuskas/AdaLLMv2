import os
import sys

from huggingface_hub import snapshot_download


def _log(message: str, quiet: bool) -> None:
    if quiet:
        return
    print(message, file=sys.stderr, flush=True)


def resolve_model_path(
    model: str,
    download_dir: str | None = None,
    verbose: bool = True,
) -> str:
    quiet = not verbose or os.getenv("ADALLM_QUIET", "").strip().lower() in ("1", "true", "yes", "on")
    if os.path.isdir(model):
        path = os.path.abspath(model)
        _log(f"Using local model path: {path}", quiet)
        return path
    try:
        path = snapshot_download(
            repo_id=model,
            cache_dir=download_dir,
            local_files_only=True,
        )
        _log(f"Using cached model from Hugging Face: {path}", quiet)
        return path
    except Exception:
        _log(f"Downloading model from Hugging Face: {model}", quiet)
        return snapshot_download(
            repo_id=model,
            cache_dir=download_dir,
            local_files_only=False,
        )
