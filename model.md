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
- checkpoint dtype: bfloat16

Include:

- token embedding
- RMSNorm
- separate q/k/v/o projections
- grouped-query attention
- RoPE
- causal self-attention
- SwiGLU FFN
- residual connections
- final RMSNorm
- untied LM head
- KV cache

Reference implementation notes:

- The PyTorch reference should support the real TinyLlama config.
- The demo may instantiate a much smaller config so it runs quickly in the local env.
- Batch size remains 1 for the C bring-up path.

Do not include yet:

- training
- tokenizer implementation
- real checkpoint loading
- quantization
- flash attention
- multi-batch serving
