# EmbeddingGemma 2 text embedder

Date: 2026-10-13. Design for review. No runtime code until this document is accepted.

Sources: [model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2), [Hugging Face `google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2), [launch post, 2026-10-06](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/), [overview](https://ai.google.dev/gemma/docs/embeddinggemma), [sentence-transformers guide](https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers), [EmbeddingGemma 1 card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card), [LiteRT text bundle](https://huggingface.co/litert-community/embeddinggemma-2-text-270m-litert-lm), [ggml-org GGUF](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF).

## 1. Background and goals

TinyInfer today is a TinyLlama decoder (`model.md`, `infer/src/llama.cc`): f32 CPU and Metal, greedy decode, persistent KV cache. EmbeddingGemma 2 is a different model and a different serving pattern.

Goal: one text-embedding entry point inside TinyInfer for `google/embeddinggemma-2`, resident on an Apple Silicon MacBook Air. The engine is purpose-built for this model. It stays small, allocates up front, and returns one vector per input.

Success: a forward matches a sentence-transformers reference, then int8 and int4 weights stay inside the Air memory caps, then a benchmark table records latency, throughput, and footprint.

## 2. Non-goals

- A generic graph runtime, a second Llama, or a multimodal tower (vision 170M, audio 300M). Those towers are optional on the card and stay unloaded.
- Tokenizer, sampler, chat template, or training.
- Retargeting `LlamaInferEngine` or `load_llama_config_json`.
- Float16 activations or weights. The model card says float16 goes NaN or silently degrades.
- ONNX Runtime, Core ML, or TensorRT as dependencies.
- Persistent KV cache. This call does not generate tokens.

## 3. Target hardware

### Primary: MacBook Air, Apple Silicon

RAM is not recorded in the repo. Two unified-memory budgets, not a Max or Ultra:

- 8 GB: TinyInfer process under about 2 GB RSS.
- 16 GB: under about 4 GB RSS.

`sysctl hw.memsize` picks the cap. The caps are plan limits, not measurements. Do not reuse timings from `docs/metal_backend_validation.md` (M5, TinyLlama, f32) or from LiteRT's MacBook Pro M5 table.

Metal is the shipping backend. Compute stays float32, matching the kernels in `infer/src/metal_backend.mm`. The card's bfloat16 switch is `torch.cuda.is_bf16_supported()`, so the Air reference run is float32.

### Secondary: NVIDIA RTX 3060

The user confirmed "JTX3060" means a GeForce RTX 3060. Assume 12 GB VRAM. An 8 GB desktop variant exists; `nvidia-smi --query-gpu=name,memory.total --format=csv` decides which card is in the box. The 3060 does not change the Air phases, the RSS caps, or the embedder API.

CUDA considerations, if that backend is added later:

- `DeviceType` today is `cpu` and `metal` only (`infer/include/tinyinfer/core.h`). A CUDA backend would be one more `Backend` for the same `embed()` call, not a new framework.
- Ampere can run the card's preferred CUDA dtype, bfloat16. Float16 stays forbidden. Float32 remains the numerical reference.
- Same packed int8/int4 weight layout as Metal. No separate quant scheme, no TensorRT engine.
- One embed call, one synchronize. No graph capture in the first CUDA path.

int8 and int4 are viable on both VRAM sizes because the text tower is small (section 6). On 12 GB they are a shared file format and a bandwidth choice, not a capacity requirement. On the 8 GB variant they are the same files; bfloat16 text also fits. Neither card loads the 740M tower in this design.

## 4. Model architecture

### Verified

| Item | Value |
| --- | --- |
| Id, license | `google/embeddinggemma-2`, Apache 2.0 |
| Parameters | 740M total. Text 270M = 130M backbone + 140M embedder. Vision 170M. Audio 300M. |
| Stack | Gemma 4 decoder, used to embed |
| Layers / width | 24 layers, model dim 512, FFN 2048 |
| Attention | 4 heads, GQA/MQA, local KV heads 2, global KV heads 1, local:global 5:1, sliding window 1024 |
| Vocabulary | 262,144 |
| FFN | Gated FFN with GELU |
| Pool | Mean pool, then 512→768 |
| Context | 8,192 tokens |
| Output | 768-d. Matryoshka 128, 256, 512. Re-normalize after a cut. |
| Languages | 100+ at inference. Training text covers 140+. Quality is not uniform. |
| Precision | bfloat16 or float32. Not float16. |
| Hugging Face tensor type | BF16 |
| Text-only load | Leave vision and audio configs unset. Effective size 270M. |
| Task strings | Prefixes such as `task: search result \| query: …` and `title: none \| text: …` |
| Official text quant | LiteRT-LM: int4 per-channel QAT for the text transformer and the embedding table. Text bundle 165 MB. |

`ggml-org/embeddinggemma-2-GGUF` has text BF16 (558 MB) and text Q8_0 (310 MB), architecture `gemma-embedding2`. No Q4 file is in that listing. External check only.

### Inferred or derived

- Head dim 128 is derived (512 / 4). Confirm in `config.json` before a kernel hard-codes it.
- 24 layers at 5:1 would be 20 local and 4 global. The card gives the ratio, not the index list. Read the schedule from the checkpoint.
- RMSNorm and RoPE are not on this card. They are typical of Gemma. Inferred. The implementation takes the op from `config.json` and from a sentence-transformers forward.
- The attention mask is not stated. The card says both "Gemma 4 decoder" and mean-pooled embeddings. A community GGUF note claimed a bidirectional mask. That note is not the card. The mask comes from the checkpoint.
- Tensor names, biases, and tying are not on the card. The loader reads the safetensors header.

EmbeddingGemma 1 is a different model: text-only, Gemma 3, about 300M (the v2 overview also writes 308M), context 2,048, same Matryoshka widths. Id on the older ONNX tree: `google/embeddinggemma-300m`. Its tensors do not load here.

## 5. Engine design

One function: token-id rows in, one vector per row out.

Google built a Gemma 4 decoder. The serving pattern is still a single forward over a known sequence, then mean pool. Layer `i`'s K and V die when layer `i` finishes. `LlamaInferEngine` keeps a session KV arena because decode appends one token (`infer/src/llama.cc`). This path does not. Scratch for the current layer sits in the workspace:

- Local layer: K and V for the 1024-token window, 2 KV heads. Verified window and verified local head count.
- Global layer: K and V for this sequence, 1 KV head. Dropped before the next layer.

No `MemoryKind::kv_cache` arena.

Layout, two `MemoryArena`s (`infer/include/tinyinfer/core.h`):

- Weights: one read-only allocation, 64-byte aligned. Packed int8 or int4 plus a scale vector per matrix once quantized. No second buffer holding the unpacked floats.
- Workspace: sized for a length bucket, reset every call. Buckets match LiteRT's published text lengths: 128, 256, 512, 1024, 2048, 8192. Float32 slots reused across all 24 layers: two `[bucket, 512]` hidden mats, two `[bucket, 2048]` FFN mats, local KV scratch `[1024, 2, 128]`, global KV scratch `[bucket, 1, 128]`, and one `[768]` output per batch row. The 128 in that shape is the derived head dim.

Mean pool uses the true length, not the bucket pad. Batch 1 is the first correct path.

Copies:

- Phase A keeps the Hugging Face weights resident as BF16 in the weight arena. It does not allocate a second float copy of them (that copy would be about 1.08 GB). The kernels this bring-up reuses cannot read that buffer: `small_m_gemm_tiled_f32` loads weights as `const device float*` / `float4`, and `embedding_f32` loads the table as `const device float*` (`infer/src/metal_backend.mm`). CPU `matmul_out` and `embedding_out` likewise read `float*` (`infer/src/cpu_backend.cc`).
- GEMM converts BF16→f32 in the weight-tile load. Each lane already consumes 4 weight values per K step (`k_per_tile` 128). Those 4 values are converted in registers, then the existing f32 dot runs. The X tile stays the current threadgroup `float4` buffer, because activations are already f32. No device scratch and no unpacked weight arena.
- The embedding gather converts BF16→f32 while writing the existing f32 hidden row (`hidden` 512, the workspace slot from the layout above). CPU gather does the same element conversion. It does not `memcpy` a float row out of a BF16 table. No extra table buffer.
- Quantized GEMM dequantizes a tile in threadgroup memory and accumulates float32.
- Residual add is in place.
- `copy_to_host` runs once, for the pooled vectors.
- Metal already encodes every op onto one command buffer and commits at `synchronize` (`infer/src/metal_backend.mm`). One `embed()` synchronizes once.

Metal kernels, narrow:

- Bring-up keeps the `small_m_gemm_tiled_f32` tile (4×4 output, K tile 128, f32 dots) and the `embedding_f32` gather shape. It does not pass the BF16 arena to those kernels. Weight and table loads convert BF16→f32 as in the copies list. Queries are a short `M` with `N` in {512, 2048}.
- A large-`M` GEMM waits until a profile shows the small-`M` kernel bandwidth-bound at long sequences.
- int8/int4 GEMM is a sibling kernel, not a mode flag on the f32 kernel.
- Local attention reads at most 1024 keys. Global attention reads this sequence. Mask mode is an argument after the checkpoint names it.
- Mean pool reuses the shape of the existing threadgroup reduction, then the 512→768 projection.
- Fusion waits on a profile. The first candidate is the gated GELU. Norm fusion waits until the norm op is verified.

Reuse: arenas, `TensorView`, `Status`, Metal device and command-buffer flush, CPU f32 matmul and embedding gather as the reference shape, tiled GEMM, `ctest`.

Leave untouched: `LlamaInferEngine`, `infer/src/llama.cc`, `load_llama_config_json` (it requires `model_type=llama`, `hidden_act=silu`, and `lm_head`), SwiGLU, RoPE, causal attention, argmax, the Llama KV layout, and `tests/test_tiny_llama.py`.

New code is an embedder, its loader, and its kernels. `CMakeLists.txt` gains those sources only.

## 6. Memory budget

Byte counts below are `parameters × width` or products of verified shapes. They are weight or scratch sizes, not RSS.

| Store | Text 270M |
| --- | --- |
| float32 weights | ~1.08 GB |
| bfloat16 weights | ~540 MB |
| int8 weights, before scales | ~270 MB |
| int4 weights, before scales | ~135 MB |
| Token table, bf16 (262144 × 512 × 2) | 256 MB |

Workspace at 8192 tokens is dominated by the FFN slot, about 128 MB (`2 × 8192 × 2048 × 4`). At 512 tokens it is a few megabytes. Weights set the Air budget.

- 16 GB Air: bfloat16 storage plus float32 compute fits the 4 GB RSS cap for bring-up. int8 is the comfortable resident format if int4 loses too much cosine.
- 8 GB Air: bfloat16 is a short check only. Resident format is int4, under the 2 GB RSS cap.
- RTX 3060, 12 GB assumed: text bfloat16, int8, and int4 all fit. Plan cap for the text process is 4 GB of VRAM, leaving the rest of the card free. The 8 GB variant uses a 3 GB plan cap for the same text path. Both caps are plan limits, not measurements.

## 7. Quantization

Activations stay float32 on Metal. On CUDA, bfloat16 compute is allowed by the card; float16 is not. Activation quantization is out of scope.

Weight order:

1. Bring-up: Hugging Face BF16 stored, float32 compute on the Air. BF16→f32 happens in the GEMM tile load and the embedding gather (section 5), not in a second weight buffer. Accuracy reference.
2. int8 per output channel. A zero-point exists only if the file has one. Symmetric is not assumed. ggml Q8_0 is an external cosine check, not the on-disk format.
3. int4 per output channel, matching verified LiteRT text QAT. Order: token table, FFN gate/up/down (512×2048), then Q/K/V/O. No invented group size.

Load QAT scales when a file ships them. A homemade calibration set is allowed only after the unquantized forward matches, and its cosine is reported on its own.

The 3060 uses these same packed tensors. int4 is viable there and is not required for a 12 GB fit.

## 8. Benchmark and acceptance criteria

Warm up one call. Record machine id (`hw.memsize` and chip on the Air; `nvidia-smi` name and VRAM on the 3060).

Air grid (primary):

| Axis | Points |
| --- | --- |
| Batch | 1, 4, 8 |
| Sequence | 128, 512, 2048. 8192 only if the RSS cap holds. |
| Width | 768 and 256 (truncate, re-normalize) |
| Weights | BF16-stored f32 compute, int8, int4 |

Each cell: p50 and p95 of `embed()` in milliseconds, sequences/s, tokens/s, peak RSS, weight-arena bytes, workspace-arena bytes, finite outputs, cosine against the frozen reference.

3060 rows (secondary, same prompts and the same weight files):

| Card | Weights | Batch | Sequence | Record |
| --- | --- | --- | --- | --- |
| 12 GB, assumed | bf16 compute, int8, int4 | 1, 4, 8 | 128, 512, 2048, 8192 | p50, p95, tokens/s, `nvidia-smi` memory, cosine |
| 8 GB variant | same | 1, 4, 8 | 128, 512, 2048 | same; 8192 only inside the 3 GB plan cap |

Reference: sentence-transformers, text tower, float32, same token ids. Prompts: the model-card pair (northern lights / charged particles) plus 32 fixed texts added when implementation starts.

Acceptance for this project, not figures from Google:

- BF16-stored and int8: cosine at least 0.999 vs the float32 reference, zero NaNs.
- int4: zero NaNs, cosine published per sequence length. It is not required to hit 0.999. If it falls below 0.99 on the 32-text set, the 16 GB Air and the 12 GB 3060 keep int8 resident; the 8 GB Air uses int4 only when the 2 GB cap requires it.
- RSS under 2 GB (8 GB Air) or 4 GB (16 GB Air) at batch 8 and the longest enabled sequence. 3060 text process under 4 GB VRAM (12 GB card) or 3 GB VRAM (8 GB card).

## 9. Phased roadmap

The Air is the phase clock. Each phase ends before the next. Existing Llama `ctest` stays green. The 3060 rows run after phase D, with the weights phase D already accepted. They are not a phase of their own.

### A — run

CPU, then Metal. Text weights stay resident BF16. Float32 compute. GEMM converts BF16→f32 in the weight-tile load (4 values per lane, registers only). Gather converts BF16→f32 into the existing f32 hidden row. No unpacked weight arena. Batch 1, sequences 128 and 512. Mask, norm, and position op from the checkpoint. Output 768-d, L2-normalized.

Verify: cosine at least 0.999 on the prompt pair and the 32 texts. Finite outputs. Llama tests pass. RSS recorded. An 8 GB Air may miss the 2 GB cap here; that does not block a 16 GB Air, and it is why phase B exists.

### B — quantize

int8 per-channel Metal GEMM, in-tile dequant, float32 accumulate. Then int4 in the section 7 order.

Verify: int8 meets 0.999 and the RSS cap. int4 meets the NaN check, the published-cosine rule, and the RSS cap. Workspace bytes do not grow.

### C — optimize

Only kernels the profile shows as hot. Expected order: local-window attention, then large-`M` GEMM if 2048 is bandwidth-bound, then one fused GELU-gate if the unfused FFN is hottest. Still one command buffer per call.

Verify: cosine vs phase B for the same weights at least 0.999999. p50 at batch 1, sequence 512, does not regress. RSS does not regress.

### D — benchmark

Fill the Air grid. Store the raw log beside the binary, not in this file.

Verify: every cell has latency, throughput, RSS, arena bytes, and cosine. A cell over its cap or under its cosine gate fails the phase. Llama smokes still match their previous token ids.

## 10. Risks and open questions

- Mask, norm, and position encoding are not verified on the card. A wrong guess silently wrecks cosine.
- Head dim and the local/global schedule are derived. `config.json` can disagree.
- int4 cosine on this implementation is unknown until phase B. The 0.99 floor is a project rule.
- Air RAM (8 vs 16) and 3060 VRAM (12 vs 8) are assumed until the machine commands in section 3 are run.
- QAT scale layout inside a file TinyInfer can parse is not verified. Phase B may have to calibrate.
- The C++ engine has no tokenizer. Callers pass token ids. A mismatched tokenizer makes a false cosine failure.
- Metal bfloat16 compute is not assumed. CUDA bfloat16 is allowed by the card and still needs a NaN check against the float32 reference.

## 11. Alternatives considered

- Loading the checkpoint through `load_llama_from_hf_dir`. Rejected. The loader requires Llama, SiLU, and `lm_head`.
- Keeping a persistent KV cache sized like `build_memory_plan`. Rejected. There is no decode loop. Scratch for one layer is enough.
- Core ML or an ONNX float16 graph as the Air runtime. Rejected. Third-party conversions use float16, which the card forbids. Community ONNX q8/q4 files stay external checks.
- TensorRT on the 3060. Rejected for this design. 12 GB is not short of capacity, and an FP16 engine conflicts with the card.
- A generic op dispatcher that also serves TinyLlama. Rejected. The Llama path stays as it is.
- Group-size int4 other than per-channel. Not used. LiteRT's verified scheme is per-channel. A group size would be an invention.
