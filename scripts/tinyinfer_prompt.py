from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "TinyLlama-1.1B-Chat-v1.0"
DEFAULT_RUNNER = REPO_ROOT / "build" / "run_token_ids_smoke"
BOS_TOKEN_ID = 1


def load_tokenizer(model_dir: Path):
    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"missing tokenizer.json: {tokenizer_path}")

    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Python package 'tokenizers' is required. "
            "Run this through `./scripts/tinyinfer` or `uv run --project reference ...`."
        ) from exc

    return Tokenizer.from_file(str(tokenizer_path))


def encode_prompt(tokenizer, prompt: str, add_bos: bool) -> list[int]:
    token_ids = list(tokenizer.encode(prompt).ids)
    if add_bos and (not token_ids or token_ids[0] != BOS_TOKEN_ID):
        token_ids.insert(0, BOS_TOKEN_ID)
    return token_ids


def parse_token_line(output: str, label: str) -> list[int]:
    prefix = f"{label}:"
    for line in output.splitlines():
        if line.startswith(prefix):
            rest = line[len(prefix) :].strip()
            return [] if not rest else [int(item) for item in rest.split()]
    raise RuntimeError(f"C++ runner did not print {label}")


def ensure_runner(runner: Path, build: bool) -> None:
    if runner.is_file():
        return
    if not build:
        raise FileNotFoundError(f"missing C++ runner: {runner}")

    print("building C++ runner...", file=sys.stderr)
    subprocess.run(
        ["cmake", "--build", str(REPO_ROOT / "build"), "--target", "run_token_ids_smoke"],
        cwd=REPO_ROOT,
        check=True,
    )
    if not runner.is_file():
        raise FileNotFoundError(f"missing C++ runner after build: {runner}")


def run_cpp(
    runner: Path,
    model_dir: Path,
    max_seq_len: int,
    max_new_tokens: int,
    prompt_tokens: list[int],
) -> str:
    cmd = [
        str(runner),
        str(model_dir),
        str(max_seq_len),
        str(max_new_tokens),
        *(str(token) for token in prompt_tokens),
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout, file=sys.stderr, end="")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        raise RuntimeError("C++ inference runner failed")
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TinyInfer from a text prompt.")
    parser.add_argument("prompt", nargs="*", help="Prompt text. If omitted, read from stdin.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--no-bos", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    prompt = " ".join(args.prompt).strip()
    if not prompt:
        prompt = input("prompt> ").strip()
    if not prompt:
        print("empty prompt", file=sys.stderr)
        return 1

    ensure_runner(args.runner, build=not args.no_build)
    tokenizer = load_tokenizer(args.model_dir)
    prompt_tokens = encode_prompt(tokenizer, prompt, add_bos=not args.no_bos)
    if len(prompt_tokens) + args.max_new_tokens > args.max_seq_len:
        print("prompt is too long for max_seq_len", file=sys.stderr)
        return 1

    print("TinyInfer is thinking on CPU f32...")
    runner_output = run_cpp(
        args.runner,
        args.model_dir,
        args.max_seq_len,
        args.max_new_tokens,
        prompt_tokens,
    )

    generated_tokens = parse_token_line(runner_output, "generated_tokens")
    all_tokens = parse_token_line(runner_output, "all_tokens")
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    all_text = tokenizer.decode(all_tokens, skip_special_tokens=True)

    print()
    print(all_text)

    if args.verbose:
        print()
        print("prompt_tokens:", " ".join(str(token) for token in prompt_tokens))
        print("generated_tokens:", " ".join(str(token) for token in generated_tokens))
        print("generated_text:", generated_text)
        print()
        print(runner_output, end="")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
