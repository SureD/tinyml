# Design principles

## This project

- Lightweight and minimal. One model gets a purpose-built path. EmbeddingGemma 2 does not become a generic engine, and it does not absorb the TinyLlama decoder.
- Memory-optimal for a MacBook Air: plan arenas up front, no persistent KV cache for embeddings, packed int8 then int4, no extra full-size float copy of the weights.
- Docs before code. [docs/embeddinggemma2-design.md](../docs/embeddinggemma2-design.md) is the review gate.
- Label facts. **Verified** means a model card, a Hugging Face file listing, or a file in this repo. **Inferred** means a Gemma-family habit or a count derived from verified numbers. Do not fill gaps with invented shapes, group sizes, or benchmark results.

## Already in the repo

`AGENTS.md` is the standing engineering taste for runtime code: correctness before speed, simple before clever, explicit ownership, low allocation on hot paths, flat layout, benchmarkable changes, small C-style APIs. `docs/arch.md` applies that to TinyInfer: `MemoryArena` owns memory, `TensorView` does not, the backend owns the kernels, execution is eager.
