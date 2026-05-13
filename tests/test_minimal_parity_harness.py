from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from array import array
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_ref_python(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["./ref/.venv/bin/python", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


class MinimalParityHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp_dir = tempfile.TemporaryDirectory()
        cls.golden_dir = Path(cls._temp_dir.name) / "golden"
        result = _run_ref_python("tools/export_demo_case.py", "--output-dir", str(cls.golden_dir))
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout)
        cls.case_path = cls.golden_dir / "case.json"
        cls.expected_logits_path = cls.golden_dir / "expected_logits.bin"

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp_dir.cleanup()

    def _load_case(self) -> dict[str, object]:
        return json.loads(self.case_path.read_text(encoding="utf-8"))

    def test_exporter_creates_complete_case(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            export_dir = Path(temp_dir) / "golden"
            result = _run_ref_python("tools/export_demo_case.py", "--output-dir", str(export_dir))
            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)

            self.assertEqual(
                json.loads((export_dir / "case.json").read_text(encoding="utf-8")),
                self._load_case(),
            )
            self.assertEqual(
                json.loads((export_dir / "weights_manifest.json").read_text(encoding="utf-8")),
                json.loads((self.golden_dir / "weights_manifest.json").read_text(encoding="utf-8")),
            )

            exported_files = sorted(
                path.relative_to(export_dir)
                for path in export_dir.rglob("*")
                if path.is_file()
            )
            committed_files = sorted(
                path.relative_to(self.golden_dir)
                for path in self.golden_dir.rglob("*")
                if path.is_file()
            )
            self.assertEqual(exported_files, committed_files)

            for rel_path in committed_files:
                self.assertEqual((export_dir / rel_path).read_bytes(), (self.golden_dir / rel_path).read_bytes())

    def test_compare_accepts_oracle_output(self) -> None:
        case = self._load_case()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            copied_logits = temp_dir_path / "actual_logits.bin"
            shutil.copyfile(self.expected_logits_path, copied_logits)
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

            result = _run_ref_python("tools/compare_demo_result.py", str(result_json), "--case", str(self.case_path))
            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            self.assertIn("PASS:", result.stdout)

    def test_compare_rejects_token_change(self) -> None:
        case = self._load_case()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            copied_logits = temp_dir_path / "actual_logits.bin"
            shutil.copyfile(self.expected_logits_path, copied_logits)
            bad_tokens = list(case["expected_generated_tokens"])
            bad_tokens[-1] += 1
            result_json = temp_dir_path / "result.json"
            result_json.write_text(
                json.dumps({"generated_tokens": bad_tokens, "logits_path": copied_logits.name}),
                encoding="utf-8",
            )

            result = _run_ref_python("tools/compare_demo_result.py", str(result_json), "--case", str(self.case_path))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("generated_tokens mismatch", result.stderr)

    def test_compare_rejects_wrong_logit_size(self) -> None:
        case = self._load_case()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            bad_logits = temp_dir_path / "bad_logits.bin"
            data = self.expected_logits_path.read_bytes()
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

            result = _run_ref_python("tools/compare_demo_result.py", str(result_json), "--case", str(self.case_path))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("logits size mismatch", result.stderr)

    def test_compare_rejects_tolerance_mismatch(self) -> None:
        case = self._load_case()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            bad_logits = temp_dir_path / "bad_logits.bin"
            values = array("f")
            values.frombytes(self.expected_logits_path.read_bytes())
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

            result = _run_ref_python("tools/compare_demo_result.py", str(result_json), "--case", str(self.case_path))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("logits mismatch", result.stderr)


if __name__ == "__main__":
    unittest.main()
