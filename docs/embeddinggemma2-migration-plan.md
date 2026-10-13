# EmbeddingGemma 2 on TinyInfer for a MacBook Air

Date: 2026-10-13. Design only. No runtime code in this change.

Goal: run EmbeddingGemma 2 text embeddings inside TinyInfer on an Apple Silicon MacBook Air, smoothly enough to stay resident. The engine is purpose-built for this one model. It is not a second Llama, and it is not a generic graph runtime.

The Air's RAM is not recorded in the repo. Plan for two unified-memory budgets, and do not assume a Max or Ultra:

- 8 GB Air: keep the TinyInfer process under about 2 GB RSS, so macOS and the editor still fit.
- 16 GB Air: keep it under about 4 GB RSS.

`sysctl hw.memsize` decides which budget applies. These RSS figures are plan caps, not measurements. Do not reuse latency from `docs/metal_backend_validation.md` (Apple M5, TinyLlama, f32) or from LiteRT's MacBook Pro M5 table.

Text only. Vision (170M) and audio (300M) stay unloaded. The launch post and the LiteRT cards say those towers are optional.

One caveat: a machine the user called "JTX3060" is not an NVIDIA product name. If that box is ever used, treat a confirmed 12 GB desktop RTX 3060 as a secondary CUDA check. It does not drive this design.

## 1. Architecture

Sources: [model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2), [Hugging Face `google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2), [launch post, 2026-10-06](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/), [overview](https://ai.google.dev/gemma/docs/embeddinggemma), [sentence-transformers guide](https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers).

### Verified

| Item | Value |
| --- | --- |
| Id and license | `google/embeddinggemma-2`, Apache 2.0 |
| Parameters | 740M total. Text 270M = 130M backbone + 140M embedder. Vision 170M. Audio 300M. |
| Stack | Gemma 4 decoder, used as an embedding model |
| Layers | 24 |
| Model dimension | 512 |
| FFN hidden dimension | 2048 |
| Heads | 4 |
| KV heads | 2 local, 1 global |
| Local:global | 5:1 |
| Attention | GQA / MQA |
| Sliding window | 1024 tokens |
| Vocabulary | 262,144 |
| Activation | Gated FFN with GELU |
| Pool | Mean pool, then a 512→768 projection |
| Context | 8,192 tokens, shared across modalities |
| Output | 768-d. Matryoshka prefixes 128, 256, 512. Re-normalize after a cut. |
| Languages | 100+ at inference. Training text covers 140+ languages. Quality is not uniform. |
| Precision | bfloat16 or float32. Not float16. The card says float16 activations go NaN or silently degrade. |
| Checkpoint dtype on Hugging Face | BF16 |
| Text-only load | Skip vision and audio (`vision_config` and `audio_config` unset). Effective size 270M. |
| Task text | Prefixes such as `task: search result \| query: …` and `title: none \| text: …` |

Official text quantization, from the LiteRT-LM cards Google's launch post points at: int4 per-channel QAT for the text transformer and the embedding table. Download size for the text bundle is 165 MB ([`litert-community/embeddinggemma-2-text-270m-litert-lm`](https://huggingface.co/litert-community/embeddinggemma-2-text-270m-litert-lm)). Vision on the larger bundles is int8 per-channel QAT. That vision scheme is out of scope until a tower is loaded.

`ggml-org/embeddinggemma-2-GGUF` publishes a text BF16 file (558 MB) and a text Q8_0 file (310 MB), architecture `gemma-embedding2`. No Q4 file is in that listing. Useful as an external numerical check, not as the TinyInfer format.

### Derived, inferred, or not on the card

- Head dimension 128 is derived: model dimension 512 divided by 4 heads. Confirm from `config.json` before freezing a kernel.
- A 5:1 schedule across 24 layers would be 20 local layers and 4 global layers. The card states the ratio, not the layer index list. Read the schedule from the checkpoint.
- RMSNorm and RoPE are not named on this card. They are typical of the Gemma line. Treat both as inferred. The first implementation reads the real op from `config.json` and from a sentence-transformers forward. It does not hard-code a Llama RMSNorm or Llama RoPE call and hope.
- The attention mask is not stated. The card calls the stack a Gemma 4 decoder and also says the output is a mean-pooled embedding. A community GGUF readme claimed bidirectional attention. That claim is not the model card. The mask comes from the checkpoint.
- Per-tensor names, biases, and whether any matrix is tied are not in the card. The loader learns them from the safetensors header.

### Predecessor, labeled so it is not mixed in

EmbeddingGemma 1 ([model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card)) is text-only, Gemma 3, about 300M (the v2 overview also writes 308M), context 2,048, same 768-d Matryoshka cuts. Its QAT note is mixed int4/int8, plus Q8_0 and Q4_0. Hugging Face id on the older ONNX tree: `google/embeddinggemma-300m`. None of those tensors load here.

## 2. Engine design

One type, one entry point: embed a batch of token-id rows, return one vector per row. No sampler, no chat template, no modality switch.

The serving pattern is an encoder even though Google built the weights as a Gemma 4 decoder. The call is a single forward over a known token sequence, then mean pool. Nothing is generated, so nothing from step N is reread at step N+1.

### KV cache

No persistent KV cache.

`LlamaInferEngine` keeps K and V for every layer for the whole session because decode appends one token at a time (`infer/src/llama.cc` plans `kv_cache_bytes` across all layers). An embedding call does not append. K and V for layer `i` die when layer `i` finishes.

Scratch for the current layer still exists, inside the workspace arena:

- Local layer: K and V for the sliding window only (1024 positions, 2 KV heads). Verified window and verified local KV-head count.
- Global layer: K and V for this sequence only (1 KV head). Discarded before the next layer.

That scratch is activation memory, not a cache. There is no `MemoryKind::kv_cache` arena on this path.

### Quantization

Activations stay float32. The model card forbids float16, and TinyInfer's kernels are already f32 (`infer/src/cpu_backend.cc`, `infer/src/metal_backend.mm`). Do not quantize activations in this design.

Weights, in order:

1. Bring-up stores the Hugging Face BF16 checkpoint and computes in float32. Text BF16 is about 540 MB (`270e6 × 2`). The token table alone is 262,144 × 512 × 2 = 256 MB. This fits the 16 GB cap with room, and it is the accuracy reference on the Air. On an 8 GB Air it is only a short-lived check: macOS plus a 540 MB table plus the process overhead can press the 2 GB cap, so the 8 GB machine does not stay on BF16.
2. int8 per-channel is the first quantized Metal kernel. One scale (and a zero-point only if the tensor file has one) per output row. Symmetric is not assumed. Text int8 weights are about 270 MB before scales. Q8_0 from ggml-org is an external cosine check, not the file format TinyInfer stores.
3. int4 per-channel is the resident format, matching the verified LiteRT text QAT scheme (transformer and embedding table). About 135 MB of packed weights before scales. Order inside that step: token table first (half the text footprint), then the FFN gate/up/down matrices (the other large GEMMs at 512×2048), then Q/K/V/O. Per-output-channel, not an invented group size.

If a LiteRT or safetensors QAT file ships scales, load those scales. A homemade calibration set is a fallback only after the unquantized forward matches sentence-transformers, and its cosine is reported rather than assumed equal to int8.

8 GB Air runs int4 text day to day. 16 GB Air may keep int8 if int4 loses more cosine than the accuracy gate allows. Neither machine loads the 740M tower.

### Memory layout

Two arenas, both from the existing `MemoryArena` (`infer/include/tinyinfer/core.h`).

**Weights.** One allocation, read-only after load, 64-byte aligned tensors (same slack idea as `build_memory_plan` in `infer/src/llama.cc`). Quantized tensors are packed int8 or int4 plus a small scale vector beside each matrix. No second buffer that holds the unpacked float matrix.

**Workspace.** Sized once for a sequence bucket, reset every `embed()` call. Buckets follow the lengths LiteRT already publishes for text: 128, 256, 512, 1024, 2048, 8192. A request uses the smallest bucket that fits. Float32 slots, reused across all 24 layers:

- two hidden mats, `[bucket, 512]`, ping-pong
- two FFN mats, `[bucket, 2048]`
- one local KV scratch, `[1024, 2, 128]`, and one global KV scratch, `[bucket, 1, 128]`
- one output row, `[768]`, per batch element

Derived head dim is in that scratch shape. If `config.json` disagrees, the scratch changes and the kernel changes with it.

At 512 tokens the workspace is a few megabytes. At 8192 the FFN slot dominates and is still on the order of 100 MB (`2 × 8192 × 2048 × 4` bytes), far under either RSS cap. Weights decide the budget, not the workspace. There is no third arena.

Batching pads each row up to the bucket. Mean pool uses the true length, not the pad. Batch 1 is the first correct path. The benchmark later measures 4 and 8.

### Copies

- The loader writes packed weights straight into the weight arena. It does not decode BF16 into a temporary full-size float tensor and then copy.
- Token gather writes hidden state into the workspace. Token ids stay in a small host or Metal buffer. Rows of the table are not staged on the CPU.
- int8 and int4 GEMM dequantizes a tile into threadgroup memory, then FMAs in float32. The unpacked matrix is never stored.
- Residual add is in place on the hidden ping-pong buffer.
- The GPU returns one 768-d vector (or the chosen Matryoshka prefix, already re-normalized) per row. `copy_to_host` runs once per `embed()` call.
- Metal already records every op onto one command buffer and commits it at `synchronize` (`infer/src/metal_backend.mm`). One `embed()` call encodes the whole forward and synchronizes once. No per-layer commit.

### Metal

Keep the kernels narrow.

- Unquantized and BF16-weight path: reuse the existing small-M tiled f32 GEMM (`small_m_gemm_tiled_f32`). Embedding queries are a short `M` (batch × tokens) with `N` in {512, 2048}. That is the shape that kernel was built for.
- Long documents make `M` large. A large-M tile is a later kernel, written only after a profile shows the small-M kernel bandwidth-bound at 2048 or 8192. Do not guess that tile size here.
- Quantized GEMM is a sibling kernel, not a flag inside the f32 one: load packed bytes, dequant a panel to float in the threadgroup, same FMA loop, float32 accumulate.
- Local attention reads at most 1024 keys. Global attention reads this sequence. Four-head, head-dim-128 attention is a short kernel with a float32 online softmax. The mask mode is a kernel argument once the checkpoint says what it is.
- Mean pool is a reduction over the valid tokens, same shape of threadgroup reduction the Metal RMSNorm kernel already uses, writing 512 floats and then the 512→768 projection.
- Fusion waits on a profile. Likely first fuse is the gated GELU (gate, up, multiply, down's input) if it shows up as extra bandwidth. Norm fusion waits until the norm op is verified.

Apple GPU on an Air is bandwidth-bound at this width. Occupancy tricks matter less than not re-reading the token table and not materializing 24 layers of KV.

### What to reuse, what to leave alone

Reuse:

- `MemoryArena`, `TensorView`, `Status` / `Result`, `DeviceType::metal`
- Backend allocation, host/device copy, and the single command-buffer flush
- CPU f32 matmul and embedding gather as the obvious shape of a reference kernel
- Metal tiled GEMM and the reduction pattern
- CMake target style and `ctest`

Leave the Llama decoder untouched:

- `LlamaInferEngine`, `infer/src/llama.cc`, `infer/include/tinyinfer/llama.h`
- `load_llama_config_json` and the Llama tensor list in `infer/src/model_loader.cc` (it requires `model_type=llama`, `hidden_act=silu`, and `lm_head`)
- `swiglu_out`, `rope_inplace`, causal `attention_out`, `argmax`, greedy prefill/decode
- The persistent KV layout `[layer, kv_head, position, head_dim]`
- `tests/test_tiny_llama.py` and the existing C++ smokes

New code lives beside that path: an embedder type, its own config loader, and its own kernels. `CMakeLists.txt` gains sources only. No ONNX Runtime, Core ML, or TensorRT dependency. Core ML conversions published by third parties use float16, which this model card forbids.

## 3. Benchmark plan

Run on the Air that will actually ship. Record `hw.memsize`, chip name, and macOS version in the log. Warm up one untimed call, then time.

| Axis | Points |
| --- | --- |
| Batch | 1, 4, 8 |
| Sequence | 128, 512, 2048. Add 8192 only if the RSS cap still holds. |
| Output width | 768 and 256 (truncate and re-normalize) |
| Weight format | BF16-stored f32 compute, then int8, then int4 |

Report, each cell:

- Latency: p50 and p95 of `embed()`, milliseconds
- Throughput: sequences per second and tokens per second
- Footprint: peak RSS, plus weight-arena bytes and workspace-arena bytes from the counters already on `MemoryArena`
- Correctness: every output finite, shape matches, cosine against a frozen sentence-transformers reference

Reference: sentence-transformers, text tower only, float32 on MPS or CPU (the published dtype switch keys bfloat16 off CUDA, so the Air reference is float32). Same token ids, not a second tokenizer's retokenization, once TinyInfer has the ids. Prompts include the model-card pair ("What causes the northern lights?" / the charged-particle sentence) plus a fixed set of 32 texts checked into the repo later.

Accuracy gates for this project, not numbers from Google:

- BF16-stored and int8 paths: cosine at least 0.999 versus the float32 reference on that set, zero NaNs.
- int4: zero NaNs, and the cosine is published per sequence length. It is not required to hit 0.999. The ONNX community card reports text q4 around 0.988 versus fp32; that is their graph, not ours. If our int4 cosine falls below 0.99 on the 32-text set, keep int8 as the resident format on the 16 GB Air and only use int4 on the 8 GB Air if the RSS cap demands it.

RSS gates: under 2 GB on an 8 GB Air, under 4 GB on a 16 GB Air, measured at batch 8 and the longest sequence that phase enables.

## 4. Roadmap

Each phase ends before the next starts. Llama `ctest` stays green throughout.

### Phase A — running

CPU forward, then Metal, text only, BF16 weights, float32 compute, batch 1, sequences 128 and 512. Mask, norm, and position op taken from the checkpoint. Output 768-d, L2-normalized.

Verify: cosine at least 0.999 against sentence-transformers on the prompt pair and the 32-text set. Finite outputs. Existing Llama tests pass. RSS recorded. An 8 GB Air may fail the 2 GB cap here; that does not block phase A on a 16 GB Air, and it is the reason phase B exists.

### Phase B — quantize

int8 per-channel Metal GEMM with in-tile dequant and float32 accumulate. Then int4 per-channel in the order in section 2: token table, FFN, attention. Load QAT scales if the file has them.

Verify: int8 meets the 0.999 cosine gate and the RSS cap for this Air. int4 meets the NaN check, the published-cosine rule, and the RSS cap. Workspace bytes do not grow versus phase A.

### Phase C — optimize

Only kernels the phase A/B profile shows as hot. Expected order: local-window attention that does not read past 1024 keys, then the large-M GEMM if 2048-token prompts are bandwidth bound, then one fused GELU-gate if the unfused FFN is the top kernel. Still one command buffer per call.

Verify: cosine unchanged versus the phase B output for the same weight format (bitwise or cosine above 0.999999). p50 latency at batch 1, sequence 512, does not regress. RSS does not regress.

### Phase D — benchmark validation

Fill the table in section 3. Store the raw log next to the binary output, not in this doc.

Verify: every cell has latency, throughput, RSS, arena bytes, and cosine. Any cell over the RSS cap or under its cosine gate fails the phase. 8192 is reported only when it fits. The Llama smoke binaries still match their previous token ids.

## Sources

- https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2
- https://ai.google.dev/gemma/docs/embeddinggemma
- https://ai.google.dev/gemma/docs/embeddinggemma/model_card
- https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers
- https://huggingface.co/google/embeddinggemma-2
- https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/
- https://huggingface.co/litert-community/embeddinggemma-2-text-270m-litert-lm
- https://huggingface.co/ggml-org/embeddinggemma-2-GGUF
