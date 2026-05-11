from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from array import array
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "golden"
CASE_PATH = GOLDEN_DIR / "case.json"
EXPECTED_LOGITS_PATH = GOLDEN_DIR / "expected_logits.bin"


def _run_ref_python(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "--project", "ref", "python", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _load_case() -> dict[str, object]:
    return json.loads(CASE_PATH.read_text(encoding="utf-8"))


class MinimalParityHarnessTests(unittest.TestCase):
    def test_exporter_regenerates_committed_case(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            export_dir = Path(temp_dir) / "golden"
            result = _run_ref_python("tools/export_demo_case.py", "--output-dir", str(export_dir))
            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)

            self.assertEqual(
                json.loads((export_dir / "case.json").read_text(encoding="utf-8")),
                _load_case(),
            )
            self.assertEqual(
                json.loads((export_dir / "weights_manifest.json").read_text(encoding="utf-8")),
                json.loads((GOLDEN_DIR / "weights_manifest.json").read_text(encoding="utf-8")),
            )

            exported_files = sorted(
                path.relative_to(export_dir)
                for path in export_dir.rglob("*")
                if path.is_file()
            )
            committed_files = sorted(
                path.relative_to(GOLDEN_DIR)
                for path in GOLDEN_DIR.rglob("*")
                if path.is_file()
            )
            self.assertEqual(exported_files, committed_files)

            for rel_path in committed_files:
                self.assertEqual((export_dir / rel_path).read_bytes(), (GOLDEN_DIR / rel_path).read_bytes())

    def test_compare_accepts_oracle_output(self) -> None:
        case = _load_case()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            copied_logits = temp_dir_path / "actual_logits.bin"
            shutil.copyfile(EXPECTED_LOGITS_PATH, copied_logits)
            result_json = temp_dir_path / "result.json"
            result_json.write_text(
                json.dumps(
                    {
                        "generated_tokens": case["expected_generated_tokens"],
                        "logits_path": copied_logits.name,
                    }
                ),
                encoding="utf-8",
            )

            result = _run_ref_python("tools/compare_demo_result.py", str(result_json))
            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            self.assertIn("PASS:", result.stdout)

    def test_compare_rejects_token_change(self) -> None:
        case = _load_case()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            copied_logits = temp_dir_path / "actual_logits.bin"
            shutil.copyfile(EXPECTED_LOGITS_PATH, copied_logits)
            bad_tokens = list(case["expected_generated_tokens"])
            bad_tokens[-1] += 1
            result_json = temp_dir_path / "result.json"
            result_json.write_text(
                json.dumps({"generated_tokens": bad_tokens, "logits_path": copied_logits.name}),
                encoding="utf-8",
            )

            result = _run_ref_python("tools/compare_demo_result.py", str(result_json))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("generated_tokens mismatch", result.stderr)

    def test_compare_rejects_wrong_logit_size(self) -> None:
        case = _load_case()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            bad_logits = temp_dir_path / "bad_logits.bin"
            data = EXPECTED_LOGITS_PATH.read_bytes()
            bad_logits.write_bytes(data[:-4])
            result_json = temp_dir_path / "result.json"
            result_json.write_text(
                json.dumps(
                    {
                        "generated_tokens": case["expected_generated_tokens"],
                        "logits_path": bad_logits.name,
                    }
                ),
                encoding="utf-8",
            )

            result = _run_ref_python("tools/compare_demo_result.py", str(result_json))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("logits size mismatch", result.stderr)

    def test_compare_rejects_tolerance_mismatch(self) -> None:
        case = _load_case()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            bad_logits = temp_dir_path / "bad_logits.bin"
            values = array("f")
            values.frombytes(EXPECTED_LOGITS_PATH.read_bytes())
            values[0] += 0.01
            with bad_logits.open("wb") as handle:
                values.tofile(handle)

            result_json = temp_dir_path / "result.json"
            result_json.write_text(
                json.dumps(
                    {
                        "generated_tokens": case["expected_generated_tokens"],
                        "logits_path": bad_logits.name,
                    }
                ),
                encoding="utf-8",
            )

            result = _run_ref_python("tools/compare_demo_result.py", str(result_json))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("logits mismatch", result.stderr)


if __name__ == "__main__":
    unittest.main()
