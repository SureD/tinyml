from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from reference.tiny_llama import REQUIRED_MODEL_FILES, TinyLlama, TinyLlamaConfig, model_files_ready


class TinyLlamaReferenceTests(unittest.TestCase):
    def test_demo_forward_shape(self) -> None:
        torch.manual_seed(0)
        model = TinyLlama(TinyLlamaConfig.demo()).eval()
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

        with torch.no_grad():
            logits = model(input_ids)

        self.assertEqual(tuple(logits.shape), (1, 4, model.config.vocab_size))

    def test_demo_generate_preserves_prompt_and_adds_tokens(self) -> None:
        torch.manual_seed(0)
        model = TinyLlama(TinyLlamaConfig.demo()).eval()
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

        with torch.no_grad():
            generated = model.generate(input_ids, max_new_tokens=3)

        self.assertEqual(tuple(generated.shape), (1, 7))
        self.assertEqual(generated[:, : input_ids.size(1)].tolist(), input_ids.tolist())

    def test_model_files_ready_requires_full_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            self.assertFalse(model_files_ready(model_dir))

            for file_name in REQUIRED_MODEL_FILES:
                (model_dir / file_name).write_text("", encoding="utf-8")

            self.assertTrue(model_files_ready(model_dir))


if __name__ == "__main__":
    unittest.main()
