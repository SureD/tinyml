# TinyInfer Architecture

TinyInfer is a dedicated LLaMA-style inference engine. It is not a generic ML graph framework. The first target is to make the TinyLlama inference path readable, testable, and easy to port to optimized Metal kernels.

## Core Objects

- `MemoryArena` owns one large backend allocation. It is move-only and releases the whole block through RAII.
- `TensorView` is a non-owning view into a `MemoryArena`. It records dtype, shape, strides, and byte offset. It never frees memory.
- `Device` names where an arena lives. v1 only distinguishes CPU and Metal.
- `Backend` owns arena allocation, host/device copies, operator implementation, and synchronization. v1 does not expose a public `Stream`; the backend owns its default execution queue internally.
- `LlamaInferEngine` is the public model API. It owns the arenas, model tensor views, KV cache views, and inference flow.
- The internal model state stores `LlamaConfig` plus tensor views for embedding, attention, MLP, final norm, and LM head.
- The internal KV cache stores decode state with tensor layout `[layer, kv_head, position, head_dim]` for both keys and values.

## Memory Model

The project uses static memory planning rather than per-tensor allocation. Tensor views are carved from arenas during initialization or workspace reset.

`LlamaInferEngine::create()` plans and allocates the long-lived arenas:

- `weights`: model weights, loaded once and treated as read-only during inference.
- `kv_cache`: persistent request/session state.
- `workspace`: temporary activation memory reused by prefill/decode.

The core rule is:

```text
plan -> allocate arenas -> bind TensorView descriptors -> run -> destroy context
```

There is no public memory planning function and no individual tensor free. `TensorView` destruction does nothing. `MemoryArena` destruction releases the complete allocation.

## Boundary

The model is specialized; the backend is flexible. This keeps the code simple without locking the runtime to one implementation of tensor math.

`LlamaInferEngine` should read like the model equation:

1. Embed token ids.
2. For each layer, run RMSNorm, QKV projection, RoPE, attention, output projection, RMSNorm, SwiGLU MLP, and residual updates.
3. Run final RMSNorm and LM head.
4. During prefill, write all prompt positions into the KV cache.
5. During decode, append one KV position and select the next token greedily.

The current skeleton records the API and high-level call order. Real kernels, scratch-buffer planning, embedding lookup, residual adds, and KV writes will be filled in later.

## Why This Shape

The central API language is `TensorView`, not a model graph and not an owning tensor object. That keeps memory ownership explicit and makes shape/layout operations cheap.

The backend owns operation implementations rather than the tensor view. This keeps `TensorView` small and lets CPU, Metal, and future backends share the same model-side inference flow.

The engine does not use lazy evaluation in v1. MLX-style lazy graphs are powerful, but this project is focused on learning and debugging the inference stream. Explicit execution makes memory ownership, synchronization, and kernel order easier to inspect.

The public API does not expose `Stream` in v1. A single backend-owned execution queue is enough while the runtime supports one device and one request. A stream abstraction can be added later for concurrent requests, copy/compute overlap, or CUDA-like backends.

## Out Of Scope For v1

- Tokenizer integration
- Hugging Face checkpoint loading
- Quantization
- Batch size greater than 1
- Continuous batching
- Paged attention
- Lazy graph scheduling
- Public stream/event dependency management
- Hidden system allocation in the decode hot path
- CPU/Metal parity kernels beyond the API skeleton
