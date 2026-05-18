# TinyInfer Architecture

TinyInfer is a dedicated LLaMA-style inference engine. It is not a generic ML graph framework. The first target is to make the TinyLlama inference path readable, testable, and easy to port to optimized Metal kernels.

## Core Objects

- `Tensor` is the common data object, inspired by MLX `array`. It records dtype, shape, strides, device, and storage. It does not know anything about LLaMA and it does not execute operations.
- `Storage` owns the backend allocation through RAII. It is move-only so tensor ownership stays explicit.
- `Device` names where data lives. v1 only distinguishes CPU and Metal.
- `Stream` controls execution order for a backend. For Metal this will map to command queue / command buffer state.
- `Backend` owns allocation, host/device copies, and operator implementations. Backend polymorphism is only used at coarse operator boundaries, where kernel launch or full-tensor work dominates virtual call overhead.
- `LlamaModel` is intentionally model-specific. It stores `LlamaConfig` plus named tensor weights for embedding, attention, MLP, final norm, and LM head.
- `KVCache` is model-specific decode state. Its planned layout is `[layer, kv_head, position, head_dim]` for both keys and values.
- `LlamaRunner` owns the readable inference flow: prefill, decode, and generate.

## Boundary

The model is specialized; the backend is flexible. This keeps the code simple without locking the runtime to one implementation of tensor math.

`LlamaRunner` should read like the model equation:

1. Embed token ids.
2. For each layer, run RMSNorm, QKV projection, RoPE, attention, output projection, RMSNorm, SwiGLU MLP, and residual updates.
3. Run final RMSNorm and LM head.
4. During prefill, write all prompt positions into the KV cache.
5. During decode, append one KV position and select the next token greedily.

The current skeleton records the API and high-level call order. Real kernels, scratch-buffer planning, embedding lookup, residual adds, and KV writes will be filled in later.

## Why This Shape

The central API language is `Tensor`, not a model graph. That gives model code, test code, and backend code one shared object to talk about.

The backend owns operation implementations rather than the tensor. This keeps `Tensor` small and lets CPU, Metal, and future backends share the same model-side inference flow.

The engine does not use lazy evaluation in v1. MLX-style lazy graphs are powerful, but this project is focused on learning and debugging the inference stream. Explicit execution makes memory ownership, synchronization, and kernel order easier to inspect.

## Out Of Scope For v1

- Tokenizer integration
- Hugging Face checkpoint loading
- Quantization
- Batch size greater than 1
- Continuous batching
- Paged attention
- Lazy graph scheduling
- Hidden allocation in the decode hot path
- CPU/Metal parity kernels beyond the API skeleton
