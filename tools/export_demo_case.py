from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ref.tiny_llama import TinyLlama, TinyLlamaConfig


DEFAULT_OUTPUT_DIR = REPO_ROOT / "golden"
SEED = 0
PROMPT_TOKENS = [1, 2, 3, 4]
MAX_NEW_TOKENS = 4


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = tensor.detach().cpu().contiguous().numpy().astype(np.dtype("<f4"), copy=False)
    array.tofile(path)


def export_demo_case(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    torch.manual_seed(SEED)
    model = TinyLlama(TinyLlamaConfig.demo()).eval()
    prompt = torch.tensor([PROMPT_TOKENS], dtype=torch.long)

    with torch.no_grad():
        logits = model(prompt)
        generated_tokens = model.generate(prompt, max_new_tokens=MAX_NEW_TOKENS).squeeze(0).tolist()

    manifest_entries: list[dict[str, object]] = []
    for name in sorted(model.state_dict()):
        rel_path = Path("weights") / f"{name}.bin"
        _write_tensor(output_dir / rel_path, model.state_dict()[name].to(torch.float32))
        manifest_entries.append(
            {
                "name": name,
                "shape": list(model.state_dict()[name].shape),
                "dtype": "float32",
                "path": rel_path.as_posix(),
            }
        )

    expected_logits_path = Path("expected_logits.bin")
    _write_tensor(output_dir / expected_logits_path, logits.to(torch.float32))

    _write_json(output_dir / "weights_manifest.json", {"tensors": manifest_entries})
    _write_json(
        output_dir / "case.json",
        {
            "seed": SEED,
            "prompt_tokens": PROMPT_TOKENS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "config": asdict(model.config),
            "expected_generated_tokens": generated_tokens,
            "expected_logits_path": expected_logits_path.as_posix(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the minimal PyTorch demo case.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the golden case into.",
    )
    args = parser.parse_args()

    export_demo_case(args.output_dir.resolve())
    print(f"exported demo case to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
