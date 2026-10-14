# Workflow

Docs only until the user accepts [docs/embeddinggemma2-design.md](../docs/embeddinggemma2-design.md). Implementation starts after that review, not before.

Then four phases, in order. A phase finishes before the next one starts.

1. **Run.** Text forward, BF16 weights, float32 compute, CPU then Metal, batch 1.
2. **Quantize.** int8 per-channel, then int4 per-channel. Activations stay float32 on Metal.
3. **Optimize.** Only kernels a profile shows as hot.
4. **Benchmark.** Latency, throughput, footprint, cosine. The RTX 3060 rows are after this phase, not a fifth phase that changes the Air plan.

Every phase keeps the existing Llama `ctest` green (`tinyinfer_tests` in `CMakeLists.txt`, plus the checkpoint and token-id smokes). New tests sit beside those binaries. They do not weaken `tests/test_tiny_llama.py`.

The Llama files stay on the Llama path: `infer/src/llama.cc`, `load_llama_config_json`, SwiGLU, causal attention, argmax, persistent KV. The embedder is new sources on `tinyinfer`.
