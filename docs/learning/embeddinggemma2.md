# EmbeddingGemma 2 learning log

Checked on 2026-10-13 against `google/embeddinggemma-2` `config.json`, the `model.safetensors` header, and the transformers text model (`modeling_embedding_gemma2.py` / `configuration_embedding_gemma2.py` on main).

## Architecture that went into the code

These are the values the CPU embedder uses. Each one was read from a file, not filled in from the model card's rounded summary.

- Local layers (the 20 `sliding_attention` entries): head dim 256, 4 query heads, 2 KV heads. `q_proj` is `[1024, 512]`.
- Global layers 5, 11, 17, and 23 (`full_attention`, keys `"05"`, `"11"`, `"17"`, `"23"`): head dim 512, 4 query heads, 1 KV head. `q_proj` is `[2048, 512]`.
- The card's "head dim 128" was `512 / 4`. `config.json` disagrees. The checkpoint agrees with 256 and 512.
- Sliding window is an inclusive radius of 512: a query attends where `abs(q - k) <= 512`. The card's 1024 is the width that radius describes. The `+1` in the attention module is only the flash-attention inclusive bound. The eager mask does not add it.
- Attention is bidirectional on every layer. There is no causal mask and no KV cache.
- RMSNorm, epsilon `1e-6`. Every norm has a weight except `v_norm`, which has none in the checkpoint.
- RoPE is the default rotate-half form. Theta is 10000 on sliding layers and 1000000 on full layers. The inverse frequencies use that layer's head dim.
- Attention scores are multiplied by 1. The model sets `scaling = 1.0`. It is not `1/sqrt(head_dim)`.
- The FFN is a gated tanh-GELU: `down(gelu(gate(x)) * up(x))`. Activation name: `gelu_pytorch_tanh`.
- Per-layer embeddings are projection-only. `inputs_embeds` (already scaled by `sqrt(hidden_size)`) go through one matrix, times `1/sqrt(hidden_size)`, then RMSNorm. Each layer gates that slice back into the residual. `embed_scale` is not a tensor; the checkpoint's 413 `language_model.*` tensors do not include it.
- The text model projects each token from 512 to 768. Sentence-transformers then mean-pools and L2-normalizes (`modules.json`, `1_Pooling`, `2_Normalize`). Truncation to 128, 256, or 512 keeps the leading entries and normalizes again.
- Float16 is rejected. Storage in the official file is bfloat16. Compute here is float32.

Not checked yet:

- Cosine against sentence-transformers on the official checkpoint. Callers pass token ids; this tree has no tokenizer.
- Metal kernels and Apple RSS. This machine is not Apple Silicon.
- The LiteRT int4 QAT file layout. Per-channel is the verified granularity. The byte packing is not.

## Phase A — run

CPU only. `reference/embeddinggemma2.py` is the float32 twin of the C++ forward. `tests/data/embeddinggemma2_tiny` is a 2-layer bfloat16 checkpoint generated from that twin. `embedder_tests` loads it and compares both sequences. The gate is cosine at least 0.999999 and max absolute error at most `2e-4`.

Measured on this Linux machine, both sequences: cosine `1`, max absolute error `0`. That is a bit match against the Python twin on this libm, not a sentence-transformers score on `google/embeddinggemma-2`.

No kernel was changed for speed. The matmul is a straight `M-N-K` loop so a mismatch stays easy to see. Llama `ctest` is still the same binary.

### What to read before changing a kernel

Nothing below was applied. Use it when a profile of `embed()` says which kernel is hot.

- **Arithmetic intensity / roofline.** The text tower is a stack of small-`M` GEMMs (`M` = sequence length, `K`/`N` in {512, 2048, 1024, 2048}). Read a roofline note for the chip you are on (Apple GPU: Metal shader profiler; NVIDIA: Nsight Compute). Try a blocked GEMM only after the profile shows the triple loop is bandwidth-bound or launch-bound, and re-check cosine.
- **Memory hierarchy and Apple unified memory.** Weights should stay in one arena. A second full float copy of a quantized matrix blows the Air RSS cap even when the math is right. On a discrete GPU the same copy also burns a transfer. Read the arena layout in `infer/src/embedder.cc` before adding a buffer.
- **Coalescing and tiling / threadgroup memory.** The design's quantized GEMM dequantizes a tile in threadgroup memory and accumulates float32. Read Apple's threadgroup-memory GEMM samples, or CUDA's tiled GEMM, before writing that kernel. Do not guess a tile size.
- **Occupancy and fusion.** One `embed()` should stay one command buffer. Fusing the GELU gate is the first candidate only if the unfused FFN shows up as the hot span. Norm fusion waits until you are sure the RMSNorm you are fusing is this one, including the unscaled `v_norm`.
- **Quantization tradeoffs.** int8 then int4, per output row, float32 accumulate. A group size other than the whole row is not in the LiteRT description. Publish cosine; do not assume int4 holds 0.999.

## Attention softmax

The first version recomputed `Q·K` and `exp` inside the loop over the head dimension. Softmax does not depend on V, so that repeated the same length-`L` weight vector `d` times: `O(L²·d²)` instead of `O(L²·d)`. `attention` now stores one score per key, turns that vector into softmax weights once, and then mixes V. That buffer is one `max_seq_len` float tensor in the workspace arena, reused by every layer. A failed allocation returns Status from init. It is not a heap vector inside `embed()`. On the tiny fixture the cosine against the frozen `expected.f32` is still 1. Max absolute error is 1.78814e-7 and 1.93715e-7, up from 0, because `exp` is no longer repeated per output dim.

**Arithmetic intensity.** The wasted work was arithmetic, not a memory round trip: each extra dot reread K that was already in registers or L1. On a GPU the same mistake also blows bandwidth, because K is reloaded from device memory per output channel.

What to read: the three passes in `attention` in `infer/src/embedder.cc`. The GPU form that does not materialize an `L×L` matrix is online softmax (Milakov and Gimelshein) and FlashAttention.

What to try: on Metal, keep one query block's softmax weights in threadgroup memory and read V coalesced. Do not allocate the full attention matrix. Re-check cosine against this CPU path before trusting a fused kernel.

## Phase B and later

Filled in on the phase branch that lands the work.
