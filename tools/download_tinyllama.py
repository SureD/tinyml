from __future__ import annotations

import argparse
from pathlib import Path


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def download_model(*, model_id: str, model_dir: Path, hf_home: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required. Run `uv sync --project ref` first."
        ) from exc

    model_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        snapshot_download(
            repo_id=model_id,
            local_dir=model_dir,
            cache_dir=hf_home,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "model.safetensors",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
            ],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TinyLlama weights into a local ignored dir.")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    args = parser.parse_args()

    path = download_model(
        model_id=args.model_id,
        model_dir=args.model_dir.resolve(),
        hf_home=args.hf_home.resolve(),
    )
    print(f"model snapshot ready: {path}")


if __name__ == "__main__":
    main()
