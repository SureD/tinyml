from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "TinyLlama-1.1B-Chat-v1.0"
DEFAULT_RUNNER = REPO_ROOT / "build" / "bench_infer"
BOS_TOKEN_ID = 1


@dataclass(frozen=True)
class BenchCase:
    name: str
    prompt_len: int
    max_seq_len: int
    max_new_tokens: int
    seed: int


CASES = (
    BenchCase("p16_d8", 16, 128, 8, 0xC0FFEE10),
    BenchCase("p64_d8", 64, 128, 8, 0xC0FFEE40),
    BenchCase("p128_d8", 128, 256, 8, 0xC0FFEE80),
)


def load_vocab_size(model_dir: Path) -> int:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing config.json: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    vocab_size = payload.get("vocab_size")
    if not isinstance(vocab_size, int) or vocab_size <= BOS_TOKEN_ID + 1:
        raise RuntimeError("config.json has invalid vocab_size")
    return vocab_size


def make_prompt_tokens(case: BenchCase, vocab_size: int) -> list[int]:
    if case.prompt_len <= 0:
        raise RuntimeError(f"invalid prompt_len for case {case.name}")

    # Token values do not affect TinyInfer kernel shapes. Use a fixed LCG so the
    # prompt is deterministic while still touching different embedding rows.
    usable_tokens = vocab_size - 3
    state = case.seed & 0xFFFFFFFF
    tokens = [BOS_TOKEN_ID]
    while len(tokens) < case.prompt_len:
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        tokens.append(3 + (state % usable_tokens))
    return tokens


def ensure_runner(runner: Path, build: bool) -> None:
    if runner.is_file():
        return
    if not build:
        raise FileNotFoundError(f"missing C++ benchmark runner: {runner}")

    build_dir = REPO_ROOT / "build"
    if not (build_dir / "CMakeCache.txt").is_file():
        print("configuring CMake...", file=sys.stderr)
        subprocess.run(
            ["cmake", "-S", str(REPO_ROOT), "-B", str(build_dir)],
            cwd=REPO_ROOT,
            check=True,
        )

    print("building bench_infer...", file=sys.stderr)
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "bench_infer"],
        cwd=REPO_ROOT,
        check=True,
    )
    if not runner.is_file():
        raise FileNotFoundError(f"missing C++ benchmark runner after build: {runner}")


def run_case(
    runner: Path,
    model_dir: Path,
    backend: str,
    case: BenchCase,
    prompt_tokens: list[int],
) -> dict[str, str]:
    cmd = [
        str(runner),
        str(model_dir),
        backend,
        str(case.max_seq_len),
        str(case.max_new_tokens),
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
        raise RuntimeError(f"bench_infer failed for case {case.name}")

    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    if len(rows) != 1:
        raise RuntimeError(f"bench_infer printed unexpected CSV for case {case.name}")

    row = rows[0]
    row["case"] = case.name
    row["seed"] = f"0x{case.seed:08x}"
    return row


def selected_cases(names: list[str] | None) -> list[BenchCase]:
    if not names:
        return list(CASES)
    wanted = set(names)
    return [case for case in CASES if case.name in wanted]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run TinyInfer's fixed benchmark case suite."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--backend", default="cpu", choices=("cpu", "metal"))
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        choices=[case.name for case in CASES],
        help="Run one fixed case. May be passed multiple times. Defaults to all cases.",
    )
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()

    try:
        ensure_runner(args.runner, build=not args.no_build)
        vocab_size = load_vocab_size(args.model_dir)
        cases = selected_cases(args.cases)

        fieldnames = [
            "case",
            "seed",
            "backend",
            "prompt_len",
            "max_seq_len",
            "max_new_tokens",
            "prefill_ms",
            "decode_total_ms",
            "decode_ms_per_token",
            "tokens_per_sec",
        ]
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()

        for case in cases:
            print(
                f"running {case.name}: prompt_len={case.prompt_len}, "
                f"max_new_tokens={case.max_new_tokens}",
                file=sys.stderr,
            )
            prompt_tokens = make_prompt_tokens(case, vocab_size)
            row = run_case(
                args.runner,
                args.model_dir,
                args.backend,
                case,
                prompt_tokens,
            )
            writer.writerow({name: row[name] for name in fieldnames})
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
