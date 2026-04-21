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