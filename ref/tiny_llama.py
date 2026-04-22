from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TinyLLaMAConfig:
    num_layers: int = 2
    hidden_size: int = 128
    num_heads: int = 4
    head_dim: int = 32
    ffn_dim: int = 256
    vocab_size: int = 256
    max_seq_len: int = 128
    batch_size: int = 1
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.hidden_size != self.num_heads * self.head_dim:
            raise ValueError(
                "hidden_size must equal num_heads * head_dim "
                f"({self.hidden_size} != {self.num_heads} * {self.head_dim})"
            )
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even to apply RoPE")


@dataclass
class KVCache:
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    seq_len: int = 0

    @classmethod
    def allocate(
        cls,
        config: TinyLLaMAConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "KVCache":
        shape = (
            config.batch_size,
            config.num_heads,
            config.max_seq_len,
            config.head_dim,
        )
        keys = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.num_layers)]
        values = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.num_layers)]
        return cls(keys=keys, values=values, seq_len=0)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.size(-1) // 2]
    x2 = x[..., x.size(-1) // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TinyLLaMAConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.max_seq_len = config.max_seq_len
        self.qkv_proj = nn.Linear(
            config.hidden_size,
            3 * config.num_heads * config.head_dim,
            bias=False,
        )
        self.out_proj = nn.Linear(
            config.num_heads * config.head_dim,
            config.hidden_size,
            bias=False,
        )

        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        positions = torch.arange(config.max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        angles = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        layer_idx: int,
        start_pos: int,
        kv_cache: KVCache | None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if start_pos + seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence would exceed max_seq_len={self.max_seq_len}: "
                f"{start_pos} + {seq_len}"
            )

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        positions = torch.arange(start_pos, start_pos + seq_len, device=x.device)
        cos = self.rope_cos.index_select(0, positions).to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        sin = self.rope_sin.index_select(0, positions).to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        total_kv_len = start_pos + seq_len
        if kv_cache is not None:
            kv_cache.keys[layer_idx][:, :, start_pos:total_kv_len, :] = k
            kv_cache.values[layer_idx][:, :, start_pos:total_kv_len, :] = v
            k = kv_cache.keys[layer_idx][:, :, :total_kv_len, :]
            v = kv_cache.values[layer_idx][:, :, :total_kv_len, :]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        query_positions = torch.arange(start_pos, total_kv_len, device=x.device)
        key_positions = torch.arange(total_kv_len, device=x.device)
        causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(
            ~causal_mask.unsqueeze(0).unsqueeze(0),
            torch.finfo(attn_scores.dtype).min,
        )

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(attn_scores.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    def __init__(self, config: TinyLLaMAConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.ffn_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.ffn_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_dim, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderBlock(nn.Module):
    def __init__(self, config: TinyLLaMAConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.ffn = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        *,
        layer_idx: int,
        start_pos: int,
        kv_cache: KVCache | None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.attn_norm(x),
            layer_idx=layer_idx,
            start_pos=start_pos,
            kv_cache=kv_cache,
        )
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TinyLLaMA(nn.Module):
    def __init__(self, config: TinyLLaMAConfig | None = None) -> None:
        super().__init__()
        self.config = config or TinyLLaMAConfig()
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderBlock(self.config) for _ in range(self.config.num_layers)]
        )
        self.final_norm = RMSNorm(self.config.hidden_size, self.config.norm_eps)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    def allocate_kv_cache(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> KVCache:
        param = next(self.parameters())
        return KVCache.allocate(
            self.config,
            device=device or param.device,
            dtype=dtype or param.dtype,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        start_pos: int = 0,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq]")
        if input_ids.size(0) != self.config.batch_size:
            raise ValueError(
                f"this reference model only supports batch_size={self.config.batch_size}"
            )

        x = self.token_embedding(input_ids)
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x, layer_idx=layer_idx, start_pos=start_pos, kv_cache=kv_cache)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        if kv_cache is not None:
            kv_cache.seq_len = max(kv_cache.seq_len, start_pos + input_ids.size(1))
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
    ) -> torch.Tensor:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        kv_cache = self.allocate_kv_cache(device=input_ids.device, dtype=self.lm_head.weight.dtype)
        logits = self(input_ids, start_pos=0, kv_cache=kv_cache)
        tokens = input_ids

        for _ in range(max_new_tokens):
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat((tokens, next_token), dim=1)
            if tokens.size(1) > self.config.max_seq_len:
                raise ValueError("generation exceeded max_seq_len")
            logits = self(next_token, start_pos=kv_cache.seq_len, kv_cache=kv_cache)

        return tokens


def _demo() -> None:
    torch.manual_seed(0)
    model = TinyLLaMA().eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    logits = model(input_ids)
    generated = model.generate(input_ids, max_new_tokens=4)
    print("logits shape:", tuple(logits.shape))
    print("generated tokens:", generated.tolist())


if __name__ == "__main__":
    _demo()
