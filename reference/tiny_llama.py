from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "TinyLlama-1.1B-Chat-v1.0"
REQUIRED_MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)


@dataclass(frozen=True)
class TinyLlamaConfig:
    # TinyLlama/TinyLlama-1.1B-Chat-v1.0 config.json values.
    num_hidden_layers: int = 22
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    vocab_size: int = 32000
    max_position_embeddings: int = 2048
    batch_size: int = 1
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5

    @classmethod
    def demo(cls) -> "TinyLlamaConfig":
        return cls(
            num_hidden_layers=2,
            hidden_size=128,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=256,
            max_position_embeddings=128,
        )

    @classmethod
    def from_hf_config(cls, path: Path) -> "TinyLlamaConfig":
        config_path = path / "config.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        return cls(
            num_hidden_layers=payload["num_hidden_layers"],
            hidden_size=payload["hidden_size"],
            intermediate_size=payload["intermediate_size"],
            num_attention_heads=payload["num_attention_heads"],
            num_key_value_heads=payload["num_key_value_heads"],
            vocab_size=payload["vocab_size"],
            max_position_embeddings=payload["max_position_embeddings"],
            rope_theta=payload.get("rope_theta", 10000.0),
            rms_norm_eps=payload.get("rms_norm_eps", 1e-5),
        )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
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
        config: TinyLlamaConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "KVCache":
        shape = (
            config.batch_size,
            config.num_key_value_heads,
            config.max_position_embeddings,
            config.head_dim,
        )
        keys = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.num_hidden_layers)]
        values = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.num_hidden_layers)]
        return cls(keys=keys, values=values)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype)
        return x * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.size(-1) // 2]
    x2 = x[..., x.size(-1) // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(x: torch.Tensor, num_repeats: int) -> torch.Tensor:
    if num_repeats == 1:
        return x
    batch_size, num_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch_size, num_kv_heads, num_repeats, seq_len, head_dim)
    return x.reshape(batch_size, num_kv_heads * num_repeats, seq_len, head_dim)


class LlamaAttention(nn.Module):
    def __init__(self, config: TinyLlamaConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        positions = torch.arange(config.max_position_embeddings, dtype=torch.float32)
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
        total_kv_len = start_pos + seq_len
        if total_kv_len > self.max_position_embeddings:
            raise ValueError(
                f"sequence would exceed max_position_embeddings={self.max_position_embeddings}: "
                f"{start_pos} + {seq_len}"
            )

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        positions = torch.arange(start_pos, total_kv_len, device=x.device)
        cos = self.rope_cos.index_select(0, positions).to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        sin = self.rope_sin.index_select(0, positions).to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            kv_cache.keys[layer_idx][:, :, start_pos:total_kv_len, :] = k
            kv_cache.values[layer_idx][:, :, start_pos:total_kv_len, :] = v
            k = kv_cache.keys[layer_idx][:, :, :total_kv_len, :]
            v = kv_cache.values[layer_idx][:, :, :total_kv_len, :]

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        query_positions = torch.arange(start_pos, total_kv_len, device=x.device)
        key_positions = torch.arange(total_kv_len, device=x.device)
        causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(
            ~causal_mask.unsqueeze(0).unsqueeze(0),
            torch.finfo(attn_scores.dtype).min,
        )

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(dtype=x.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


class LlamaMLP(nn.Module):
    def __init__(self, config: TinyLlamaConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: TinyLlamaConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = LlamaAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        *,
        layer_idx: int,
        start_pos: int,
        kv_cache: KVCache | None,
    ) -> torch.Tensor:
        x = x + self.self_attn(
            self.input_layernorm(x),
            layer_idx=layer_idx,
            start_pos=start_pos,
            kv_cache=kv_cache,
        )
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class TinyLlama(nn.Module):
    def __init__(self, config: TinyLlamaConfig | None = None) -> None:
        super().__init__()
        self.config = config or TinyLlamaConfig()
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(self.config) for _ in range(self.config.num_hidden_layers)]
        )
        self.norm = RMSNorm(self.config.hidden_size, self.config.rms_norm_eps)
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
            raise ValueError(f"this reference model only supports batch_size={self.config.batch_size}")

        x = self.embed_tokens(input_ids)
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x, layer_idx=layer_idx, start_pos=start_pos, kv_cache=kv_cache)

        x = self.norm(x)
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
            if tokens.size(1) > self.config.max_position_embeddings:
                raise ValueError("generation exceeded max_position_embeddings")
            logits = self(next_token, start_pos=kv_cache.seq_len, kv_cache=kv_cache)

        return tokens


TinyLLaMA = TinyLlama


def model_files_ready(model_dir: Path) -> bool:
    return all((model_dir / file).is_file() for file in REQUIRED_MODEL_FILES)


def ensure_hf_checkpoint(
    model_dir: Path,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    hf_home: Path | None = None,
) -> Path:
    if model_files_ready(model_dir):
        return model_dir

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download TinyLlama. "
            "Run `uv sync --project reference --locked` first."
        ) from exc

    hf_home = hf_home or Path(os.environ.get("HF_HOME", REPO_ROOT / "hf_cache"))
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"downloading model: {model_id}")
    return Path(
        snapshot_download(
            repo_id=model_id,
            local_dir=model_dir,
            cache_dir=hf_home,
            allow_patterns=list(REQUIRED_MODEL_FILES),
        )
    )


def load_hf_checkpoint(model_dir: Path, *, dtype: torch.dtype = torch.float32) -> TinyLlama:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError(
            "safetensors is required to load Hugging Face weights. "
            "Run `./scripts/test_reference.sh` to sync the reference environment."
        ) from exc

    config = TinyLlamaConfig.from_hf_config(model_dir)
    model = TinyLlama(config).to(dtype=dtype)
    checkpoint_path = model_dir / "model.safetensors"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")

    checkpoint = load_file(str(checkpoint_path), device="cpu")
    checkpoint_state = {
        name.removeprefix("model."): tensor.to(dtype=dtype)
        for name, tensor in checkpoint.items()
    }
    expected_keys = set(model.state_dict())
    allowed_extra = {
        name for name in checkpoint_state
        if name.endswith(".self_attn.rotary_emb.inv_freq")
    }
    unexpected_keys = sorted(set(checkpoint_state) - expected_keys - allowed_extra)
    if unexpected_keys:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected_keys[:5]}")

    state_dict = {
        name: tensor
        for name, tensor in checkpoint_state.items()
        if name in expected_keys
    }
    missing_keys = sorted(expected_keys - set(state_dict))
    if missing_keys:
        raise RuntimeError(f"missing checkpoint keys: {missing_keys[:5]}")

    model.load_state_dict(state_dict, strict=True)
    return model


def _demo(
    model_dir: Path | None = None,
    *,
    max_new_tokens: int = 4,
    model_id: str = DEFAULT_MODEL_ID,
    hf_home: Path | None = None,
) -> None:
    torch.manual_seed(0)
    if model_dir is None:
        config = TinyLlamaConfig.demo()
        model = TinyLlama(config).eval()
    else:
        model_dir = ensure_hf_checkpoint(model_dir, model_id=model_id, hf_home=hf_home)
        model = load_hf_checkpoint(model_dir).eval()
        config = model.config

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids)
        generated = model.generate(input_ids, max_new_tokens=max_new_tokens)
    print("model_dir:", str(model_dir) if model_dir is not None else "<demo random weights>")
    print("config:", config)
    print("logits shape:", tuple(logits.shape))
    print("generated tokens:", generated.tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TinyLlama PyTorch reference.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Local Hugging Face snapshot directory containing config.json and model.safetensors.",
    )
    parser.add_argument(
        "--model-id",
        default=os.environ.get("MODEL_ID", DEFAULT_MODEL_ID),
        help="Hugging Face model id to download when --model-dir is missing required files.",
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=Path(os.environ.get("HF_HOME", REPO_ROOT / "hf_cache")),
        help="Hugging Face cache directory used for automatic downloads.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4)
    args = parser.parse_args()
    _demo(
        model_dir=args.model_dir,
        max_new_tokens=args.max_new_tokens,
        model_id=args.model_id,
        hf_home=args.hf_home,
    )


if __name__ == "__main__":
    main()
