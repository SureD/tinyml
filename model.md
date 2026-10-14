### **Model definition**

Use the architecture from `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.

Source: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0

Core config:

- architecture: `LlamaForCausalLM`
- model type: `llama`
- hidden size: 2048
- layers: 22
- attention heads: 32
- key/value heads: 4
- head dimension: 64
- FFN intermediate size: 5632
- vocab size: 32000
- max position embeddings: 2048
- activation: SiLU / SwiGLU
- RMSNorm epsilon: 1e-5
- RoPE theta: 10000.0
- attention bias: false
- tied word embeddings: false
- checkpoint dtype: bfloat16 on disk; runtime weights are f32

Include:

- token embedding lookup (`embedding_out` on CPU and Metal)
- RMSNorm
- separate q/k/v/o projections
- grouped-query attention
- RoPE
- causal self-attention
- SwiGLU FFN
- residual connections, fused in the C++ engine with the following RMSNorm
- final RMSNorm
- untied LM head
- KV cache

The engine applies a plain RMSNorm to the embedding before layer 0. Every later residual add is `add_rms_norm_out`: it writes the sum, then RMSNorms that sum. After attention the norm weight is `post_attention_layernorm`. After the MLP it is the next layer's `input_layernorm`, or `model.norm` on the last token of the last layer. That last-token row is the input to greedy `matmul_argmax`. CPU fuses the add and norm in one loop. Metal uses `add_rms_norm_f32` when one row fits in threadgroup memory, and otherwise `add_inplace` followed by `rms_norm_out`.

Checkpoint loading, from `config.json` plus one `model.safetensors`:

- `load_llama_from_hf_dir` reads `<model_dir>/config.json` and `<model_dir>/model.safetensors`.
- `load_llama_from_hf_files` takes those two paths directly.
- Safetensors dtypes F32 and BF16 are accepted. BF16 is expanded to f32 by shifting the 16 bits into the high half of the float. Other dtypes are rejected.
- Required tensors are `model.embed_tokens.weight`, `model.norm.weight`, `lm_head.weight`, and each layer's attention projections, MLP projections, `input_layernorm`, and `post_attention_layernorm`. Shapes must match the contiguous f32 views. Any other tensor is rejected unless its name ends in `.self_attn.rotary_emb.inv_freq`.
- If present, `model_type` must be `llama`, `hidden_act` must be `silu`, `attention_bias` and `tie_word_embeddings` must be false, `pretraining_tp` must be 1, and `rope_scaling` must be null or absent. Missing or null `rms_norm_eps` and `rope_theta` default to `1e-5` and `10000.0`.

Reference implementation notes:

- The PyTorch reference should support the real TinyLlama config.
- The demo may instantiate a much smaller config so it runs quickly in the local env.
- Batch size remains 1 for the C bring-up path.

Do not include yet:

- training
- tokenizer implementation
- sharded safetensors or dtypes other than F32 and BF16
- quantization
- flash attention
- multi-batch serving
