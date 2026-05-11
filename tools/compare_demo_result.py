from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE_PATH = REPO_ROOT / "golden" / "case.json"


class CompareError(Exception):
    pass


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_logits(path: Path, *, expected_shape: tuple[int, ...]) -> np.ndarray:
    array = np.fromfile(path, dtype=np.dtype("<f4"))
    expected_numel = int(np.prod(expected_shape))
    if array.size != expected_numel:
        raise CompareError(
            f"logits size mismatch: got {array.size} floats, expected {expected_numel} "
            f"for shape {expected_shape}"
        )
    return array.reshape(expected_shape)


def compare_demo_result(
    *,
    case_path: Path,
    result_path: Path,
    atol: float,
    rtol: float,
) -> tuple[float, float]:
    case = _read_json(case_path)
    result = _read_json(result_path)

    try:
        expected_tokens = case["expected_generated_tokens"]
        prompt_tokens = case["prompt_tokens"]
        vocab_size = case["config"]["vocab_size"]
        expected_logits_rel = case["expected_logits_path"]
        actual_tokens = result["generated_tokens"]
        actual_logits_rel = result["logits_path"]
    except KeyError as exc:
        raise CompareError(f"missing required field: {exc.args[0]}") from exc

    if actual_tokens != expected_tokens:
        raise CompareError(
            "generated_tokens mismatch: "
            f"expected {expected_tokens}, got {actual_tokens}"
        )

    expected_shape = (1, len(prompt_tokens), vocab_size)
    expected_logits = _load_logits(
        _resolve_path(case_path.parent, expected_logits_rel),
        expected_shape=expected_shape,
    )
    actual_logits = _load_logits(
        _resolve_path(result_path.parent, actual_logits_rel),
        expected_shape=expected_shape,
    )

    diff = np.abs(actual_logits - expected_logits)
    max_abs_diff = float(diff.max(initial=0.0))
    denom = np.maximum(np.abs(expected_logits), 1e-12)
    max_rel_diff = float((diff / denom).max(initial=0.0))

    if not np.allclose(actual_logits, expected_logits, atol=atol, rtol=rtol):
        raise CompareError(
            "logits mismatch: "
            f"max_abs_diff={max_abs_diff:.8g}, max_rel_diff={max_rel_diff:.8g}"
        )

    return max_abs_diff, max_rel_diff


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare a C result against the golden demo case.")
    parser.add_argument("result_json", type=Path, help="Path to the result.json emitted by C.")
    parser.add_argument(
        "--case",
        type=Path,
        default=DEFAULT_CASE_PATH,
        help="Path to the golden case.json.",
    )
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance.")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance.")
    args = parser.parse_args()

    try:
        max_abs_diff, max_rel_diff = compare_demo_result(
            case_path=args.case.resolve(),
            result_path=args.result_json.resolve(),
            atol=args.atol,
            rtol=args.rtol,
        )
    except (CompareError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(
        "PASS: generated tokens match and logits are within tolerance "
        f"(max_abs_diff={max_abs_diff:.8g}, max_rel_diff={max_rel_diff:.8g})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
