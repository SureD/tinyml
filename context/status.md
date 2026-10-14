# Status and repo map

## Status

Shipped in tree: TinyLlama-1.1B-Chat f32 inference. CPU and Metal kernels, prefill, greedy decode, persistent KV cache, Hugging Face config and safetensors load into f32. `docs/metal_backend_validation.md` records token parity on an Apple M5 (2026-06-06). `model.md` still lists checkpoint loading as future work; `infer/src/model_loader.cc` has since implemented it for Llama only. The loader rejects anything that is not `model_type=llama` and `hidden_act=silu`.

Not in tree: a tokenizer inside the engine (callers pass token ids), Llama quantization, CUDA, and a Metal EmbeddingGemma kernel.

EmbeddingGemma 2 (`google/embeddinggemma-2`) is specified in [docs/embeddinggemma2-design.md](../docs/embeddinggemma2-design.md). The docs-only gate was opened for implementation. Phase A is a CPU text embedder: bfloat16 or float32 weights, float32 compute, batch 1, no KV cache. Head dim, the 5:1 layer list, RMSNorm, RoPE, and the bidirectional mask were taken from `config.json`, the safetensors header, and the transformers text model before the kernels were written. The card's derived head dim of 128 was not used. Attention softmax is computed once per query head, into one workspace buffer reused across layers. Phase B quantizes every 2-D text weight to per-row int8 or int4 on the CPU and drops the unpacked floats. The quant range includes 0 so a same-sign row does not clamp its zero point and clip. Reloading safetensors into a quantized engine is refused. One-dimensional norms stay float32. Phase C was not opened. Phase D records a Linux CPU log for the tiny fixture only: Release `-O3` `embed()` latency, sequences/s, tokens/s, arena bytes, and process RSS. That log was rerun at 2026-10-14T09:02:31Z after the score buffer moved into the workspace; workspace bytes at `max_seq_len` 64 are 32408. Metal, both MacBook Air caps, and both RTX 3060 cards are explicit not-yet-measured rows. Official-checkpoint cosine, the 0.999 gate, Apple RSS, and the LiteRT int4 file layout are not measured. Notes: [docs/learning/embeddinggemma2.md](../docs/learning/embeddinggemma2.md), log [docs/learning/embeddinggemma2-cpu-bench.log](../docs/learning/embeddinggemma2-cpu-bench.log).

## Map

| Path | What it is |
| --- | --- |
| `CMakeLists.txt` | CMake 3.20, C++20, static `tinyinfer`, `nlohmann_json`. Metal and Foundation on Apple. Tests: `tinyinfer_tests`, `load_checkpoint_smoke`, `run_token_ids_smoke`, `bench_infer`, `bench_embedder`. |
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
