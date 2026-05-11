A minimal LLaMA-style decoder-only inference engine in C + Metal on Apple Silicon.

Phase 1 focuses on correctness and interpretability:
- tiny model
- batch size 1
- KV cache
- greedy decode
- CPU reference alignment

Phase 2 focuses on GPU acceleration:
- Metal matmul
- profiling
- decode-path optimization

Minimal parity harness:
- Export the golden case with `uv run --project ref python tools/export_demo_case.py`
- Compare a future C result with `uv run --project ref python tools/compare_demo_result.py path/to/result.json`
- `result.json` must contain `generated_tokens` and `logits_path`
- Run the self-checks with `python3 -m unittest discover -s tests`
