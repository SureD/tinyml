# Status and repo map

## Status

Shipped in tree: TinyLlama-1.1B-Chat f32 inference. CPU and Metal kernels, prefill, greedy decode, persistent KV cache, Hugging Face config and safetensors load into f32. `docs/metal_backend_validation.md` records token parity on an Apple M5 (2026-06-06). `model.md` still lists checkpoint loading as future work; `infer/src/model_loader.cc` has since implemented it for Llama only. The loader rejects anything that is not `model_type=llama` and `hidden_act=silu`.

Not in tree: a tokenizer inside the engine (callers pass token ids), quantization, CUDA, and any embedding model.

EmbeddingGemma 2 (`google/embeddinggemma-2`) is specified in [docs/embeddinggemma2-design.md](../docs/embeddinggemma2-design.md). The docs-only gate was opened for implementation. Phase A is a CPU text embedder: bfloat16 or float32 weights, float32 compute, batch 1, no KV cache. Head dim, the 5:1 layer list, RMSNorm, RoPE, and the bidirectional mask were taken from `config.json`, the safetensors header, and the transformers text model before the kernels were written. The card's derived head dim of 128 was not used. Metal is not implemented. Official-checkpoint cosine, Apple RSS, and the LiteRT int4 file layout are not measured. Notes: [docs/learning/embeddinggemma2.md](../docs/learning/embeddinggemma2.md).

## Map

| Path | What it is |
| --- | --- |
| `CMakeLists.txt` | CMake 3.20, C++20, static `tinyinfer`, `nlohmann_json`. Metal and Foundation on Apple. Tests: `tinyinfer_tests`, `load_checkpoint_smoke`, `run_token_ids_smoke`, `bench_infer`. |
| `infer/include/tinyinfer/` | `core.h` (arenas, `TensorView`, `cpu`/`metal`), `backend.h`, `llama.h`, `model_loader.h`, `embedder.h`. |
| `infer/src/` | `core.cc`, `cpu_backend.cc`, `llama.cc`, `model_loader.cc`, `metal_backend.mm`, `embedder.cc`, `embedder_loader.cc`. |
| `reference/` | PyTorch TinyLlama (`tiny_llama.py`, `pyproject.toml`). |
| `tests/cpp/`, `tests/test_tiny_llama.py` | C++ smokes, the embedder reference test, and the Python Llama test. |
| `reference/embeddinggemma2.py` | Float32 twin of the EmbeddingGemma 2 text forward. |
| `scripts/` | Token-id smoke, prompt helper, bench cases. |
| `docs/arch.md` | TinyInfer architecture: arenas, eager execution, Llama equation. |
| `docs/embeddinggemma2-design.md` | EmbeddingGemma 2 design under review. |
| `model.md` | TinyLlama shape the C++ engine targets. |
| `readme.md` | Phase 0–2 for the Llama engine. |
| `AGENTS.md` | Engineering taste for runtime changes. |
| `context/` | This folder. |
