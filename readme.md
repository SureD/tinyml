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

Reference workflow:
- Download TinyLlama and run the reference smoke test with `./scripts/test_ref.sh`
- Downloaded weights live under `models/` and are ignored by git
- Export a temporary demo golden case with `./ref/.venv/bin/python tools/export_demo_case.py`
- Compare a future C result with `./ref/.venv/bin/python tools/compare_demo_result.py path/to/result.json`
- `result.json` must contain `generated_tokens` and `logits_path`
