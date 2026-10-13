# EmbeddingGemma 2 migration plan

Date: 2026-10-13. Research only. No runtime code in this change.

EmbeddingGemma 2 is a real model. It is not a drop-in replacement for the TinyLlama decoder this repo runs today. The first milestone is a text embedding that stays inside the memory of an Apple Silicon MacBook Air and a 12 GB RTX 3060. A C++ port comes after those two machines match a reference vector.

## Hardware this plan is written for

**MacBook Air, Apple Silicon.** Unified memory, not a Max or Ultra. The exact Air (M1 through M4) and its RAM are not recorded in the repo, so the plan has two budgets:

- 8 GB Air: the embedding process stays under about 2 GB RSS. macOS and the editor need the rest.
- 16 GB Air: the process stays under about 4 GB RSS.

24 GB and larger Air configs are out of the success criteria. If `sysctl hw.memsize` reports something else, re-pick the row below. Do not copy latency or RSS from the LiteRT MacBook Pro M5 table or from `docs/metal_backend_validation.md` (that run was an M5, TinyLlama, f32).

**"JTX3060".** That is not an NVIDIA product name. This plan treats it as a desktop GeForce RTX 3060, 12 GB GDDR6, Ampere, compute capability 8.6, which has native bfloat16. The 3060 family also includes an 8 GB desktop card and a 6 GB laptop card. Before any quant choice, run `nvidia-smi --query-gpu=name,memory.total --format=csv`. If the total is not 12288 MiB, use the smaller-card column. Do not assume the name was a typo for a 12 GB card without that reading.

## What EmbeddingGemma 2 is

Sources: [Google AI model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2), [Hugging Face `google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2), [launch post, 2026-10-06](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/), [overview](https://ai.google.dev/gemma/docs/embeddinggemma).

| Item | Value |
| --- | --- |
| Hugging Face id | `google/embeddinggemma-2` |
| License | Apache 2.0 |
| Parameters | 740M total. Text 270M = 130M backbone + 140M embedder. Vision 170M. Audio 300M. Towers load independently. |
| Output | 768-d, then a linear map from model dim 512. Matryoshka prefixes 128, 256, 512. Re-normalize after truncation. |
| Context | 8,192 tokens shared across modalities |
| Languages | 100+ at inference. Training text covers 140+ languages. Quality is not uniform. |
| Text shape | 24 layers, dim 512, FFN 2048, 4 heads, local/global KV heads 2/1, local:global 5:1, sliding window 1024, vocab 262,144, gated FFN with GELU, mean pool |
| Precision rule | bfloat16 or float32. Not float16. The card says float16 activations go NaN or silently degrade. |
| On-device note from Google | Quantized text weights about 191 MB active RAM, full multimodal about 567 MB, measured on a Pixel 11 Pro. Not a measurement for the Air or the 3060. |

Weight bytes below are `parameters × bytes`, or published file sizes. They are not process RSS.

| Checkpoint | Text 270M | Full 740M |
| --- | --- | --- |
| float32 weights | ~1.08 GB | ~2.96 GB |
| bfloat16 weights | ~540 MB | ~1.48 GB |
| Token table alone, vocab 262144 × 512, bf16 | 256 MB | same table |

The token table is most of the 140M embedder. A KV cache is not the long-term cost: this model is one forward plus mean pool, not a decode loop. Derived from the card (head dim 512/4 = 128, 5:1 local/global, window 1024), a full 8k forward holds on the order of a few tens of MB of K and V in bf16. Framework overhead and the embedding table dominate.

### Quantized files that were actually published

Official text QAT is int4, not int8. int8 shows up on the vision tower and in community Q8 files.

- LiteRT-LM, linked from the launch post: [`litert-community/embeddinggemma-2-text-270m-litert-lm`](https://huggingface.co/litert-community/embeddinggemma-2-text-270m-litert-lm), [`...-text-vision-440m-litert-lm`](https://huggingface.co/litert-community/embeddinggemma-2-text-vision-440m-litert-lm), [`...-740m-litert-lm`](https://huggingface.co/litert-community/embeddinggemma-2-740m-litert-lm). Text transformer and embeddings are int4 per-channel QAT. Vision is int8 per-channel QAT. Audio is mixed int2/int4/int8 per-channel QAT. Download sizes: 165 MB, 388 MB, 485 MB. Fixed text lengths include 128, 256, 512, 1024, 2048, 8192.
- llama.cpp: [`ggml-org/embeddinggemma-2-GGUF`](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF), architecture `gemma-embedding2`, Apache 2.0. Files seen: text BF16 558 MB, text Q8_0 310 MB, multimodal projector BF16 982 MB, projector Q8_0 555 MB. No Q4 file in that listing.
- ONNX: [`onnx-community/embeddinggemma-2-ONNX`](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX) (community, not a Google-owned repo). Published dtypes include fp32, fp16, q8, q4, q4f16. Their table lists text fp32 1085 MB, text q8 314 MB, text q4 175 MB, full fp32 2929 MB, full q8 850 MB, full q4 473 MB, and a worst-case cosine vs fp32 of about 0.9997 for q8 and 0.975 for q4 (text q4 about 0.988). They also list fp16 at cosine 0.9998. That fp16 file conflicts with Google's model card. Do not deploy it.
- A separate `.tflite` flatbuffer was not found. The on-device path Google points at is LiteRT-LM and MediaPipe.
- No TensorRT engine was found.
- The launch post also names transformers, sentence-transformers, MLX, vLLM, llama.cpp, SGLang, Ollama, LM Studio, and transformers.js. An official Google MLX weight repo was not verified. Core ML is not in that list. Community Core ML conversions exist and store compute in float16, which this plan rejects.

Text-only load is `config_kwargs={"vision_config": None, "audio_config": None}` in sentence-transformers (270M). Task prefixes are part of the model: `task: search result | query: ...` and `title: none | text: ...`.

### Difference from EmbeddingGemma (the first model)

The first model card ([EmbeddingGemma](https://ai.google.dev/gemma/docs/embeddinggemma/model_card), updated 2025-09-25) is text-only, Gemma 3, 300M, context 2,048, output 768 with the same 128/256/512 Matryoshka cuts, 100+ languages. Its QAT configs are mixed int4/int8, Q8_0, and Q4_0. The EmbeddingGemma 2 overview calls that predecessor EmbeddingGemma 1 and also says 308M; both figures are on Google's pages. Hugging Face id used by the older ONNX tree: `google/embeddinggemma-300m`.

EmbeddingGemma 2 adds image, video, and audio, moves the text tower from about 300M to 270M, extends context from 2k to 8k, and changes the stack to Gemma 4 (GELU gated FFN, sliding window, mixed local/global KV heads). Same native width, 768.

## What this repo is

It is a LLaMA-style decoder, not an embedding runtime.

- Model spec: `model.md`. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`. `LlamaConfig::tinyllama_1_1b()` in `infer/src/llama.cc` is 22 layers, hidden 2048, intermediate 5632, 32 heads, 4 KV heads, vocab 32000, context 2048, SiLU, RoPE. `model.md` still says tokenizer, real checkpoint loading, and quantization are out of scope. The loader has since landed; quantization has not.
- Build: `CMakeLists.txt`. CMake 3.20, C++20 and Objective-C++, static `tinyinfer`, `nlohmann_json` via FetchContent. On Apple it links Metal and Foundation. Targets: `tinyinfer_tests`, `load_checkpoint_smoke`, `run_token_ids_smoke`, `bench_infer`. No CUDA language, no ORT, no TensorRT, no Core ML.
- Layout: `infer/include/tinyinfer/{core,backend,llama,model_loader}.h`, `infer/src/{core,cpu_backend,llama,model_loader}.cc`, `infer/src/metal_backend.mm`, `reference/` (PyTorch TinyLlama, `reference/pyproject.toml` depends on torch, safetensors, tokenizers, huggingface-hub, numpy), `tests/cpp/`, `tests/test_tiny_llama.py`, `scripts/`, `docs/`.
- Devices: `DeviceType` is `cpu` and `metal` only (`infer/include/tinyinfer/core.h`). `DType` lists f16 and bf16, but the kernels and the loader are f32. `load_llama_safetensors` requires a contiguous f32 destination and returns `unimplemented` for an unsupported checkpoint dtype (`infer/src/model_loader.cc`).
- Loader gate: `model_type == llama`, `hidden_act == silu`, no rope scaling, untied embeddings, tensor names `model.embed_tokens`, per-layer `q_proj`/`k_proj`/`v_proj`/`o_proj`, `gate_proj`/`up_proj`/`down_proj`, and `lm_head`. An EmbeddingGemma 2 config fails this on purpose.
- Implemented: f32 CPU and Metal matmul, token embedding, residual, RMSNorm, RoPE, causal grouped-query attention, SwiGLU, argmax, prefill, greedy decode, KV cache layout `[layer, kv_head, position, head_dim]`. `docs/metal_backend_validation.md` records TinyLlama token parity on an M5.
- Not implemented: tokenizer inside the engine (callers pass token ids), quantization, CUDA, mean pooling, a 512→768 projection, GELU, sliding-window attention, mixed KV-head counts per layer, multimodal towers.

`LlamaInferEngine` generates token ids. EmbeddingGemma 2 returns one vector. Do not retarget `load_llama_from_hf_dir` at `google/embeddinggemma-2`.

## Runtime and quant per machine

Text only for the first ship. This repo has no image, audio, or video path, and the text tower is the piece that can share a later C++ implementation. Store 256-d Matryoshka vectors after a 768-d quality check. The card says 256-d stays close on text; 128-d is the one to re-measure. Queries and documents use the same width. Cap the first bring-up at 512 tokens. Move to 2048, then 8192, only after that machine's memory gate still holds.

| Machine | Compute | Weights | Why |
| --- | --- | --- | --- |
| Air, 8 GB | llama.cpp Metal, `gemma-embedding2` | `ggml-org` text Q8_0, 310 MB | Fits the 2 GB process budget. Q4 is not in that official GGUF listing. LiteRT int4 text (165 MB) is the fallback if Metal Q8 RSS is still too high. |
| Air, 16 GB | same Metal Q8_0 | same 310 MB file | Headroom for a 2048-token check. Still do not load the 982 MB bf16 projector. |
| RTX 3060, 12 GB confirmed | PyTorch CUDA, sentence-transformers | bf16 text tower, ~540 MB | Ampere runs bf16. The model card wants bf16 or fp32. 12 GB does not need int4. |
| NVIDIA card at 8 GB | CUDA bf16 text, or ONNX Runtime CUDA q8 text (314 MB) | no full fp32 tower | fp32 text is ~1.1 GB and still fits; full fp32 (~3 GB) plus a desktop session is the wrong default. |
| NVIDIA card at 6 GB | ONNX Runtime CUDA q8 text, or CUDA bf16 text | no multimodal tower until `nvidia-smi` memory during a 512-token forward stays under 4 GB | Laptop 3060s are 6 GB. |

Rejected on both machines:

- float16. That includes MPS fp16, a Core ML fp16 package, an ONNX `fp16` file, and a TensorRT FP16 engine. The model card forbids it. A community cosine near 1.0 does not lift the ban until a local run shows zero NaNs and matches the bf16 or fp32 reference below.
- Core ML as a target. Not in Google's runtime list, and the published conversions use float16.
- TensorRT in this migration. No official engine. The 12 GB card is not memory-bound at bf16. A later engine is in scope only for bf16 or for the LiteRT QAT dtypes, and only after the cosine gate.
- Loading `onnx-community/embeddinggemma-300m-ONNX` or `google/embeddinggemma-300m`. Those are the first model.

MPS is the PyTorch path on the Air, used only as a numerical reference: sentence-transformers, text tower, float32, sequence 512. `torch.cuda.is_bf16_supported()` is the card's bf16 switch and is false under MPS, so the documented Air PyTorch dtype is float32 (~1.1 GB weights). That is acceptable as a one-shot reference on a 16 GB Air and tight on an 8 GB Air. If the 8 GB machine swaps or the process is killed, skip MPS and compare llama.cpp Q8_0 against the 3060's saved vectors instead.

ONNX Runtime is the quantized CUDA path, not the quality path. Use the CUDA execution provider and dtype `q8`. Do not enable the fp16 graph.

## Phases

Each phase is done only if the previous gate passed. No TinyInfer code until phase 4.

### 0. Name the machines

On the Air: `sysctl machdep.cpu.brand_string hw.memsize`, and confirm Metal exists. On the NVIDIA box: `nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv`.

Gate: write the two strings into the run log. Pick the table row from the real RAM and VRAM. Stop if the GPU is not a 3060-class card at all.

### 1. Freeze a reference vector on the 3060

Environment: Python, `sentence-transformers` and `transformers`, CUDA. Load `google/embeddinggemma-2` with `torch.bfloat16` and `vision_config` / `audio_config` set to `None`.

Prompts, from the model card:

- `task: search result | query: What causes the northern lights?`
- `title: none | text: The northern lights are caused by charged particles from the sun.`

Encode at 768-d with `normalize_embeddings=True`, sequence cap 512. Save both vectors. Record `nvidia-smi` process memory.

Gate: both vectors finite, shape `(768,)`, cosine between them positive and stable across two processes. Process memory under 4 GB. Then repeat one forward at 256-d and check it matches the renormalized 768-d prefix.

### 2. Air reference, then the Metal deploy path

16 GB Air: same library, float32, MPS if `torch.backends.mps.is_available()` else CPU. Text tower only. Cosine against the 3060 bf16 vectors.

8 GB Air: skip the MPS float32 load if phase 0 showed 8 GB. Go straight to llama.cpp.

Then both Airs: current llama.cpp with `gemma-embedding2` support, Metal, `ggml-org/embeddinggemma-2-GGUF` text Q8_0, embedding mode, mean pooling, context 512. Same two prompts.

Gate: cosine versus the 3060 bf16 vectors at least 0.999 on both prompts, zero NaNs, RSS under 2 GB on the 8 GB Air and under 4 GB on the 16 GB Air. If Q8_0 misses 0.999, stop and compare BF16 GGUF (558 MB) before touching int4. If RSS fails on the 8 GB Air, switch that machine to LiteRT-LM text int4 and apply the same cosine gate. Do not "fix" a miss by casting to float16.

### 3. 3060 deploy path, and the VRAM branch

Stay on CUDA bf16 sentence-transformers for the quality path. Optionally run ONNX Runtime CUDA, text q8, same prompts.

Gate: q8 cosine versus the phase-1 bf16 vectors at least 0.999, no NaNs, `nvidia-smi` memory under 4 GB at batch 1, sequence 512. If VRAM is 8 GB or 6 GB, q8 becomes the default and bf16 must still pass the 4 GB (8 GB card) or a 3 GB (6 GB card) process limit. Do not build a TensorRT engine in this phase.

Raise context to 2048, then 8192, only on the machine whose 512-token gate passed. Record memory again. 8192 is optional. Failing it does not fail the migration.

### 4. C++ embedder, only after phases 1–3

New code, beside the Llama engine, not inside it. The first C++ target is text, f32 or bf16 storage with f32 accumulation, mean pool, 512→768, L2 norm, Matryoshka truncate. Attention mask, RoPE, and which layers are local versus global come from the checkpoint config, not from `LlamaInferEngine`'s causal mask.

Files a real port touches:

- Add `infer/include/tinyinfer/embedder.h` and `infer/src/embedder.cc`. Leave `infer/src/llama.cc` on TinyLlama.
- `infer/include/tinyinfer/backend.h`: new kernels for a GELU gated FFN, mean pool, and the Gemma attention pattern. `swiglu_out` is SiLU and is the wrong op.
- `infer/include/tinyinfer/core.h`: no dtype or device change required for a CPU f32 reference. A Metal embedder reuses `DeviceType::metal`. A CUDA backend is a new `DeviceType` and a new `.cu`, and it is not required to ship if phase 3 already serves the 3060.
- New loader. Do not extend `load_llama_config_json`. That function rejects non-`llama` and non-`silu` configs, and it requires `lm_head`.
- `CMakeLists.txt`: new sources only. No TensorRT, ONNX Runtime, or Core ML dependency.
- `reference/pyproject.toml`: add `sentence-transformers` and `transformers` for the reference script. The C++ build does not link them.
- New test beside `tests/cpp/`, cosine against the saved phase-1 vectors. Do not weaken `tests/test_tiny_llama.py`.
- `model.md` and `docs/arch.md` stay the TinyLlama documents until the embedder builds and passes. Update them in the same change as the code, not before.

Gate: CPU f32 cosine versus the saved 3060 vectors at least 0.999 on the two prompts, then the same bar for Metal on the Air at Q8-or-better only after the f32 path matches. `ctest` for the existing Llama tests still passes. No quant kernels in this phase. int4 and int8 stay in llama.cpp, LiteRT, or ONNX Runtime until an f32 C++ match exists.

## What is blocked

- Nothing about whether the model exists. The card, the Hugging Face repo, and the 2026-10-06 launch post agree.
- Casting the Llama loader at this checkpoint. It will fail closed, and working around that check would run the wrong equation.
- float16 on MPS, Core ML, ONNX, or TensorRT.
- Using Pixel, M5, or RTX 4090 LiteRT numbers as the Air or 3060 budget.
- Multimodal towers, until text passes on the actual RAM and VRAM measured in phase 0.
- An in-tree CUDA or TensorRT backend. The 3060 is served by sentence-transformers and, if needed, ONNX Runtime.

## Sources

- https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2
- https://ai.google.dev/gemma/docs/embeddinggemma
- https://ai.google.dev/gemma/docs/embeddinggemma/model_card
- https://huggingface.co/google/embeddinggemma-2
- https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/
- https://huggingface.co/litert-community/embeddinggemma-2-740m-litert-lm
- https://huggingface.co/litert-community/embeddinggemma-2-text-270m-litert-lm
- https://huggingface.co/ggml-org/embeddinggemma-2-GGUF
- https://huggingface.co/onnx-community/embeddinggemma-2-ONNX
- https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers
