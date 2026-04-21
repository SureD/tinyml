### **Model definition**

Use a **tiny LLaMA-style decoder-only Transformer** with:

- 2 layers
- hidden size 128
- 4 attention heads
- head dimension 32
- FFN dimension 256
- vocab size 256
- max sequence length 128
- batch size 1

Include:

- token embedding
- RMSNorm
- QKV projection
- RoPE
- causal self-attention
- output projection
- FFN
- residual connections
- final RMSNorm
- LM head
- KV cache

Do not include yet:

- training
- tokenizer ecosystem
- real checkpoint compatibility
- quantization
- flash attention
- multi-batch serving