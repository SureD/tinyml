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

## Phase B — quantize

CPU only. Every 2-D text weight is packed per matrix row: int8 in `[-128, 127]`, or int4 in `[-8, 7]`, two values per byte, low nibble first. Each row stores a float scale and an int8 zero point, because the checkpoint does not say the integers are symmetric. A constant row uses scale = value, zero = 0, and stored integer 1. The float matrices are read, packed, and not kept. Norms and `layer_scalar` stay float32. Activations stay float32. `gemm_quant` expands 32 weights onto the stack, then accumulates float32. That tile is a CPU stand-in. It is not a measured GPU tile size.

This packing is TinyInfer's. It is not a LiteRT QAT file, and it was not checked against one.

Measured on the tiny fixture (`tests/data/embeddinggemma2_tiny`), both sequences, Linux x86_64. Cosine is against the float32 embedder and against `expected.f32`. Those two scores matched. After the attention-score buffer moved into the workspace, `embedder_tests` on 2026-10-14 printed workspace bytes 4856 at `max_seq_len` 8, and that size did not change across quantize. Weight arena bytes are the tiny model, including 64-byte alignment slack, not the 270 MB / 135 MB estimates.

| Pack | Cosine | Max abs vs f32 | Weight arena bytes |
| --- | --- | --- | --- |
| f32 | 1 | 1.78814e-7, 1.93715e-7 vs expected.f32 | 10200 |
| int8 | 0.99984, 0.999858 | 0.013538, 0.0120243 | 10200 → 8514 |
| int4 | 0.920494, 0.820283 | 0.303526, 0.389811 | 10200 → 7610 |

The test floors are these measured scores, trimmed: int8 cosine at least 0.9998 and max abs at most 0.014; int4 cosine at least 0.820 and max abs at most 0.39. The int4 second sequence moved down from 0.846225 when the range was widened to include 0. That is this fixture, not the official 0.99 rule.

### Quantization range and the zero point

`scale = (max - min) / (qmax - qmin)` and `zero = round(qmin - min / scale)`. If every value in the row is positive, `min > 0` and that zero point is below `qmin`. The old code clamped the zero point and left the scale alone. The integer grid then covered only a slice of the row, and the large weights clipped. For int8, a row in `[1, 2]` stored both entries as 127 with zero point -128, and dequantized to `[1, 1]`. The all-negative case clamps the other way.

`quantize_matrix` now folds 0 into min and max before the scale. A positive row uses `[0, max]`, so the zero point is `qmin` and the top of the row stays representable. A negative row uses `[min, 0]`. A mixed-sign row already contains 0, so its scale does not change. An all-zero row is still the constant path.

Measured dequant of `[1, 2]` after the change: int8 `0.996078, 2`; int4 `0.933333, 2`. `[-2, -1]`: int8 `-2, -0.996078`; int4 `-2, -0.933333`.

**Quantization tradeoffs.** Including 0 spends codes on the empty side of a same-sign row. int8 on this fixture got closer to float32 (cosine 0.99984 and 0.999858). int4 spent those codes and one sequence fell from 0.846225 to 0.820283. Clipping was the worse error on a single row; the embedding cosine is a separate measurement.

What to read: the zero-point derivation in Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (the affine map `r = scale * (q - zero)`), and `quantize_matrix` in `infer/src/embedder.cc`.

What to try: keep the same-sign row test. On a GPU dequant kernel, use this scale and zero; a symmetric kernel would not match these bytes. Do not invent a group size to win the codes back.

Not measured: cosine on `google/embeddinggemma-2`, the 32-text set, sentence-transformers, RSS, and any GPU.

### Principles for this step

- **Quantization tradeoffs.** Four bits per weight is a bandwidth win and a cosine loss. On this fixture int8 stays near the float vector and int4 does not. Read the row loop in `quantize_matrix` before changing the range or the zero point. Try the official checkpoint before deciding that int4 is the resident format. Do not add a group size; the verified LiteRT granularity is the whole output row.
- **Memory hierarchy, and Apple unified memory versus a discrete GPU.** The unpacked float matrix is not resident. On a Mac the arena is the process RSS. On a discrete GPU the same second copy would be another device buffer and a transfer. Read `quantize_weights` before adding a dequantized cache.
- **Tiling / threadgroup memory.** The CPU kernel dequantizes 32 lanes on the stack so the expanded row is not a heap matrix. A Metal kernel should dequantize in threadgroup memory and accumulate float32. Read Apple's threadgroup-memory GEMM notes, and a CUDA tiled GEMM, before picking a tile. The 32 here was not profiled.
- **Arithmetic intensity / roofline.** Packed weights only help if the expanded values stay in registers or threadgroup memory. Expanding into a full float matrix drops the intensity back to the float GEMM and spends the bandwidth you just saved. Profile `embed()` at sequence 512 on the real text tower before rewriting the inner loop.

## Phase C

Not opened. No profile picked a hot kernel.

## Phase D — benchmark

Rerun at 2026-10-14T08:54:05Z after softmax-once attention and the quant range that includes 0. Release (`-O3 -DNDEBUG`, Clang 18.1.3) `bench_embedder` on this Linux host: 4 vCPUs, KVM, `Intel(R) Xeon(R) Processor`, MemTotal 16387416 kB. The model is `tests/data/embeddinggemma2_tiny` (2 layers, hidden 8, embedding dim 6), loaded with `max_seq_len` 64. Batch is 1. Each cell is one warmup `embed()` and 500 timed calls. p50 and p95 use `llround(p * (n - 1))` on the sorted samples. sequences/s is `1000 / p50`. tokens/s multiplies that by the timed sequence length. The raw log is [embeddinggemma2-cpu-bench.log](embeddinggemma2-cpu-bench.log). The 2026-10-13 numbers are not this run.

This is not `google/embeddinggemma-2` and it is not the 0.999 gate.

| Case | Quant | Seq | p50 ms | p95 ms | seq/s | tok/s | Cosine vs f32 | Cosine vs expected.f32 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixture0 | f32 | 5 | 0.007041 | 0.009341 | 142025.28 | 710126.402 | 1 | 1 |
| fixture1 | f32 | 3 | 0.004186 | 0.004273 | 238891.543 | 716674.63 | 1 | 1 |
| pattern1 | f32 | 1 | 0.001872 | 0.001914 | 534188.034 | 534188.034 | 1 | na |
| pattern8 | f32 | 8 | 0.011529 | 0.01781 | 86737.7917 | 693902.333 | 1 | na |
| pattern32 | f32 | 32 | 0.065728 | 0.071954 | 15214.2162 | 486854.917 | 1 | na |
| pattern64 | f32 | 64 | 0.198939 | 0.206345 | 5026.66647 | 321706.654 | 1 | na |
| fixture0 | int8 | 5 | 0.010052 | 0.013318 | 99482.69 | 497413.45 | 0.999840198 | 0.999840197 |
| fixture1 | int8 | 3 | 0.006165 | 0.006251 | 162206.002 | 486618.005 | 0.999858186 | 0.999858182 |
| pattern1 | int8 | 1 | 0.002804 | 0.002881 | 356633.381 | 356633.381 | 0.99996401 | na |
| pattern8 | int8 | 8 | 0.016031 | 0.016325 | 62379.1404 | 499033.123 | 0.995833747 | na |
| pattern32 | int8 | 32 | 0.082298 | 0.085549 | 12150.9636 | 388830.834 | 0.999960579 | na |
| pattern64 | int8 | 64 | 0.22919 | 0.238866 | 4363.19211 | 279244.295 | 0.999925832 | na |
| fixture0 | int4 | 5 | 0.016784 | 0.016952 | 59580.5529 | 297902.765 | 0.920493791 | 0.920493775 |
| fixture1 | int4 | 3 | 0.010252 | 0.010333 | 97541.943 | 292625.829 | 0.820283185 | 0.82028335 |
| pattern1 | int4 | 1 | 0.00409 | 0.004192 | 244498.778 | 244498.778 | 0.996864239 | na |
| pattern8 | int4 | 8 | 0.026697 | 0.03493 | 37457.3922 | 299659.138 | 0.876785306 | na |
| pattern32 | int4 | 32 | 0.124466 | 0.139499 | 8034.32263 | 257098.324 | 0.619388609 | na |
| pattern64 | int4 | 64 | 0.314508 | 0.323393 | 3179.56936 | 203492.439 | 0.798002994 | na |

Weight arena bytes: f32 10200, int8 8514, int4 7610. Workspace arena bytes: 32088 for every cell, because the arena is sized to sequence 64. Process VmRSS after the int4 loop was 4448 KiB, VmHWM 4448 KiB. The phase B test's 4760-byte workspace was the same fixture loaded with `max_seq_len` 8.

### How to read the CPU figures

Compare f32, int8, and int4 at the same sequence length. tokens/s rises with sequence length even when each call gets slower, because it counts tokens inside the call. p95 is the tail of these 500 calls, not a service latency.

int8 and int4 are slower than f32 here. The matrices are a few kilobytes, so they already sit in cache, and unpacking adds arithmetic. **Quantization tradeoffs:** packed weights save bandwidth only when the kernel is waiting on memory. **Arithmetic intensity / roofline:** this tiny tower is not that kernel. Do not carry this slowdown onto a GPU.

**Memory hierarchy.** Workspace bytes do not shrink when the timed sequence is shorter than 64. RSS here is the whole process, including code. It is not a model-footprint budget.

Pattern tokens are `1 + (i % 19)` on the tiny checkpoint. `na` means there is no frozen `expected.f32` for that length. The int4 cosine of 0.619388609 at sequence 32 is this fixture against its own float32 output. It is not the official 0.99 rule. Sequence 64 f32 p50 fell from 0.747896 ms to 0.198939 ms on this rerun because softmax is no longer repeated per head dimension. That is still this tiny model, not a GPU.

### Rows left unmeasured

The log lists every design-grid cell as `not_yet_measured`. No GPU time is filled in.

| Device | What the row is for | Result |
| --- | --- | --- |
| MacBook Air 8 GB, Metal | Unified memory. Weights and activations are the process RSS. Plan cap about 2 GB. | not yet measured |
| MacBook Air 16 GB, Metal | Same unified memory. Plan cap about 4 GB. `hw.memsize` was not run, so the cap was not chosen. | not yet measured |
| RTX 3060 12 GB | Discrete VRAM. Read `nvidia-smi`, not Linux VmRSS. A host-to-device copy is traffic the Air does not pay. Plan cap 4 GB. | not yet measured |
| RTX 3060 8 GB | Same discrete VRAM. Plan cap 3 GB. | not yet measured |

Those cells are batch 1, 4, and 8; sequence 128, 512, 2048, and 8192; width 768 and 256; weights f32, int8, and int4. This CPU run cannot stand in for them: the engine is batch 1, the fixture embedding dim is 6, and its maximum position is 64.
