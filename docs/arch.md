# TinyInfer Architecture

TinyInfer is a dedicated LLaMA-style inference engine. It is not a generic ML graph framework. It runs the TinyLlama inference path on a CPU f32 reference backend and, on Apple platforms where the Metal framework is available, on Metal f32 kernels.

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

`LlamaInferEngine::run_layers` is the inference stream:

1. Look up token ids in the embedding table (`embedding_out`).
2. RMSNorm that hidden state with layer 0 `input_layernorm`. This first norm has no residual add.
3. For each layer, project Q, K, and V, apply RoPE, and run causal grouped-query attention. Attention writes this step's K and V into the KV cache. The output projection follows.
4. Fuse each residual add with the following RMSNorm (`add_rms_norm_out`). After attention, the projected attention output is added to the incoming hidden state and normalized with `post_attention_layernorm`. The MLP is SwiGLU. On every layer except the last, the MLP output is then added to that attention residual and normalized with the next layer's `input_layernorm`.
5. On the last layer, only the last token is updated. The fused op adds the attention residual, writes final RMSNorm into that same row, and `matmul_argmax` against the untied LM head returns the greedy next token.
6. Prefill starts at position 0 and writes every prompt position. Decode appends one position.

`add_rms_norm_out` stores `residual_out = residual_out + residual`, then writes `norm_out = rms_norm(residual_out, weight)`. The CPU backend does both in one row loop. Metal runs `add_rms_norm_f32` when one row fits in threadgroup memory. If it does not, Metal calls `add_inplace` and then `rms_norm_out`. The layer loop does not call `add_inplace` itself.

CPU and Metal implement the f32 operators this stream uses: embedding, matmul, RMSNorm, fused residual RMSNorm, RoPE, KV cache writes, attention, SwiGLU, and matmul-argmax. Metal matmul dispatches `matvec_f32` when M is 1. For other M at most 16 it uses `small_m_gemm_tiled_f32` when K is a multiple of 128 and `small_m_gemm_f32` otherwise. Larger M uses `matmul_f32`. `matmul_argmax` runs `matvec_argmax_partial_f32` and then `argmax_reduce_pairs_f32`. `create_metal_backend()` returns unimplemented when the Metal framework is not available.

`load_llama_from_hf_dir` and `load_llama_from_hf_files` load one Hugging Face LLaMA checkpoint. They read `config.json` and a single `model.safetensors`, check tensor names and shapes against the engine's f32 weight views, convert F32 or BF16 values to f32, and copy them into the weight arena. BF16 expands by shifting each 16-bit value into the high half of an f32. Any other dtype is rejected. Unexpected tensor names are rejected, except keys that end in `.self_attn.rotary_emb.inv_freq`. When those config fields are present, `model_type` must be `llama`, `hidden_act` must be `silu`, `attention_bias` and `tie_word_embeddings` must be false, `pretraining_tp` must be 1, and `rope_scaling` must be null or absent. Missing or null `rms_norm_eps` and `rope_theta` default to `1e-5` and `10000.0`.

## Why This Shape

The central API language is `TensorView`, not a model graph and not an owning tensor object. That keeps memory ownership explicit and makes shape/layout operations cheap.

The backend owns operation implementations rather than the tensor view. This keeps `TensorView` small and lets CPU, Metal, and future backends share the same model-side inference flow.

The engine does not use lazy evaluation in v1. MLX-style lazy graphs are powerful, but this project is focused on learning and debugging the inference stream. Explicit execution makes memory ownership, synchronization, and kernel order easier to inspect.

The public API does not expose `Stream` in v1. A single backend-owned execution queue is enough while the runtime supports one device and one request. A stream abstraction can be added later for concurrent requests, copy/compute overlap, or CUDA-like backends.

## Out Of Scope For v1

- Tokenizer integration
- Sharded safetensors and checkpoint dtypes other than F32 and BF16
- Quantization
- Batch size greater than 1
- Continuous batching
- Paged attention
- Lazy graph scheduling
- Public stream/event dependency management
- Hidden system allocation in the decode hot path
