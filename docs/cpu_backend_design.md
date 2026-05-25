# TinyInfer CPU Backend Design

## Status

Implemented as the f32 correctness backend.

The CPU backend is intentionally simple: single-threaded scalar kernels, strict shape checks, no hidden heap allocation inside operator hot paths, and no dtype conversion. It is the numerical oracle for later Metal kernels, not the final performance target.

## Backend Boundary

`TensorView` owns no memory and never interprets backend storage. `CPUBackend` is the only code that turns a `MemoryArena` native handle into a host pointer:

```text
MemoryArena -> native CPU allocation
TensorView  -> byte offset + shape + strides
CPUBackend  -> validate view, compute pointer, run op
```

Every math op requires f32, contiguous tensors owned by the same CPU backend. Bad shapes or foreign arenas return `invalid_argument`; unsupported dtypes return `unimplemented`.

## Operator Contracts

### `matmul_out(out, x, w)`

Dense row-major linear projection:

```text
x:   [M, K]
w:   [N, K]
out: [M, N] or any contiguous shape with first dim M and M*N elements

out[m,n] = sum_k x[m,k] * w[n,k]
```

The `[N,K]` weight layout matches PyTorch/Hugging Face linear weights and avoids a loader-side transpose during early bring-up.

### `rms_norm_out(out, x, weight, eps)`

```text
x:      [T, H]
weight: [H]
out:    [T, H]
```

For each row, compute f32 `mean(x^2)`, then write `x * rsqrt(mean + eps) * weight`. In-place `out == x` is valid because the row reduction finishes before writing.

### `rope_inplace(q, k, start_pos, theta)`

```text
q: [T, H, D]
k: [T, KVH, D]
D must be even
```

RoPE uses TinyLlama/LLaMA half-split rotation:

```text
x0 = x[i]
x1 = x[i + D/2]
angle = position / theta^(2*i/D)
y[i]       = x0*cos(angle) - x1*sin(angle)
y[i + D/2] = x1*cos(angle) + x0*sin(angle)
```

`theta` is explicit in the backend API so the kernel does not hard-code model config.

### `attention_out(out, q, k, v, k_cache, v_cache, start_pos, kv_len)`

The engine passes per-layer cache views:

```text
q:       [T, H, D]
k, v:    [T, KVH, D]
k_cache: [KVH, S, D]
v_cache: [KVH, S, D]
out:     [T, H*D] or [T, H, D]
```

The op first writes current `k/v` into cache positions `[start_pos, start_pos + T)`. It then computes grouped-query causal attention over keys `[0, kv_len)`, where query head `h` maps to KV head `h / (H / KVH)`.

The CPU implementation recomputes scores during softmax instead of allocating a score buffer. That is slower but keeps the hot path allocation-free and easy to debug.

### `swiglu_out(out, gate, up)`

Elementwise:

```text
out[i] = silu(gate[i]) * up[i]
silu(x) = x / (1 + exp(-x))
```

All shapes must match exactly.

### `argmax(out_token, logits)`

Accepts `[V]` or `[1,V]` logits and returns the first maximum index. Tie behavior is deterministic.

## Engine Integration Notes

The KV arena is stored as `[layer, kv_head, position, head_dim]`. `LlamaInferEngine` derives a per-layer `[kv_head, position, head_dim]` view before calling `attention_out`, keeping layer indexing out of the backend kernel.

`LlamaInferEngine` now has custom move construction and move assignment. Moving an engine also rebinds all internal `TensorView::arena` pointers to the destination arenas; otherwise views would point at moved-from arenas and fail validation.

## Non-Goals

- No SIMD or threading yet.
- No f16/bf16 arithmetic yet.
- No Metal implementation in this step.
- No hidden temporary tensor allocation inside CPU ops.
- No generic graph or operator registry.

## Verification

The C++ test binary covers:

- CPU arena allocation, copy, and argmax.
- Matmul numerical output and shape failures.
- RMSNorm and SwiGLU numerical output.
- RoPE half-split rotation.
- Grouped-query causal attention, including KV cache writes.
- Engine move safety through the fake backend flow test.
