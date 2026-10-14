"""Float32 reference for the EmbeddingGemma 2 text tower.

The math follows transformers' EmbeddingGemma2TextModel: scaled token embeddings,
projection-only per-layer embeddings, bidirectional attention, RMSNorm, rotate-half
RoPE, and a tanh GELU gate. Attention scores are multiplied by 1, which is what
that model sets. Sliding layers keep keys with abs(q - k) <= sliding_window.

This is the numerical reference for the C++ embedder. It is not a tokenizer and
it does not load the vision or audio towers.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np

TINY_CONFIG = {
    "model_type": "embedding_gemma2",
    "dtype": "bfloat16",
    "text_config": {
        "model_type": "embedding_gemma2_text",
        "dtype": "bfloat16",
        "vocab_size": 20,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "hidden_activation": "gelu_pytorch_tanh",
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "sliding_window": 2,
        "layer_types": ["sliding_attention", "full_attention"],
        "hidden_size_per_layer_input": 4,
        "embedding_dim": 6,
        "rope_parameters": {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 1000000.0},
        },
        "per_layer_config": {
            "1": {"head_dim": 8, "num_key_value_heads": 1},
        },
    },
}

TINY_SEQUENCES = [
    [1, 4, 7, 2, 9],
    [3, 3, 8],
]


def f32(value: float) -> np.float32:
    return np.float32(value)


def f32_to_bf16(array: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(array, dtype=np.float32).view(np.uint32).copy()
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = bits + (np.uint32(0x7FFF) + lsb)
    return (rounded >> np.uint32(16)).astype(np.uint16)


def bf16_to_f32(array: np.ndarray) -> np.ndarray:
    bits = array.astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32)


def gemm(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    rows, k = x.shape
    cols = weight.shape[0]
    out = np.empty((rows, cols), dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            total = f32(0)
            for inner in range(k):
                total += x[row, inner] * weight[col, inner]
            out[row, col] = total
    return out


def rms_norm(x: np.ndarray, weight: np.ndarray | None, eps: float) -> np.ndarray:
    rows, dim = x.shape
    out = np.empty_like(x)
    eps_f = f32(eps)
    for row in range(rows):
        sum_sq = f32(0)
        for i in range(dim):
            sum_sq += x[row, i] * x[row, i]
        scale = f32(1.0) / np.sqrt(sum_sq / f32(dim) + eps_f)
        for i in range(dim):
            value = x[row, i] * scale
            if weight is not None:
                value *= weight[i]
            out[row, i] = value
    return out


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    k = math.sqrt(2.0 / math.pi)
    flat = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    out = np.empty_like(flat)
    for i, raw in enumerate(flat.tolist()):
        value = float(raw)
        inner = k * (value + 0.044715 * value * value * value)
        out[i] = f32(0.5 * value * (1.0 + math.tanh(inner)))
    return out.reshape(x.shape)


def inv_freq(theta: float, head_dim: int) -> np.ndarray:
    half = head_dim // 2
    out = np.empty(half, dtype=np.float32)
    for i in range(half):
        exponent = (2.0 * i) / float(head_dim)
        out[i] = f32(1.0 / (float(theta) ** exponent))
    return out


def apply_rope(states: np.ndarray, freq: np.ndarray) -> None:
    seq, n_heads, head_dim = states.shape
    half = head_dim // 2
    for t in range(seq):
        for h in range(n_heads):
            row = states[t, h]
            for i in range(half):
                angle = float(t) * float(freq[i])
                c = f32(math.cos(angle))
                s = f32(math.sin(angle))
                x1 = row[i]
                x2 = row[i + half]
                row[i] = x1 * c - x2 * s
                row[i + half] = x2 * c + x1 * s


def _dot(a: np.ndarray, b: np.ndarray) -> np.float32:
    total = f32(0)
    for i in range(a.shape[0]):
        total += a[i] * b[i]
    return total


def attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    sliding: bool,
    window: int,
) -> np.ndarray:
    # Softmax weights are one vector per query head. Mixing V with that vector
    # is O(seq^2 * head_dim). Recomputing Q·K per output dim was O(seq^2 * head_dim^2).
    seq, n_heads, head_dim = q.shape
    n_kv = k.shape[1]
    group = n_heads // n_kv
    out = np.empty_like(q)
    weight = np.empty(seq, dtype=np.float32)
    for t in range(seq):
        for h in range(n_heads):
            kv_head = h // group
            q_row = q[t, h]
            max_score = -math.inf
            for p in range(seq):
                if sliding and abs(t - p) > window:
                    continue
                score = float(_dot(q_row, k[p, kv_head]))
                weight[p] = f32(score)
                if score > max_score:
                    max_score = score
            max_score_f = f32(max_score)
            denom = f32(0)
            for p in range(seq):
                if sliding and abs(t - p) > window:
                    weight[p] = f32(0)
                    continue
                coeff = f32(math.exp(float(weight[p] - max_score_f)))
                weight[p] = coeff
                denom += coeff
            inv_denom = f32(1.0) / denom
            for p in range(seq):
                if sliding and abs(t - p) > window:
                    continue
                weight[p] = f32(weight[p] * inv_denom)
            for d in range(head_dim):
                total = f32(0)
                for p in range(seq):
                    if sliding and abs(t - p) > window:
                        continue
                    total += weight[p] * v[p, kv_head, d]
                out[t, h, d] = total
    return out


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    total = f32(0)
    for value in vector:
        total += value * value
    norm = np.sqrt(total)
    if float(norm) < 1e-12:
        norm = f32(1e-12)
    inv = f32(1.0) / norm
    return (vector * inv).astype(np.float32)


def layer_spec(text: dict, index: int) -> dict:
    kind = text["layer_types"][index]
    full = kind == "full_attention"
    head_dim = text["head_dim"]
    n_kv = text["num_key_value_heads"]
    override = text["per_layer_config"].get(str(index))
    if override is None:
        override = text["per_layer_config"].get(f"{index:02d}")
    if override is not None:
        head_dim = override["head_dim"]
        n_kv = override["num_key_value_heads"]
    theta = text["rope_parameters"][kind]["rope_theta"]
    return {
        "full": full,
        "head_dim": head_dim,
        "n_kv": n_kv,
        "theta": theta,
    }


def forward(text: dict, weights: dict[str, np.ndarray], tokens: list[int]) -> np.ndarray:
    hidden = text["hidden_size"]
    inter = text["intermediate_size"]
    ple_dim = text["hidden_size_per_layer_input"]
    embed_dim = text["embedding_dim"]
    n_layers = text["num_hidden_layers"]
    n_heads = text["num_attention_heads"]
    eps = float(text["rms_norm_eps"])
    window = int(text["sliding_window"])
    seq = len(tokens)
    embed_scale = f32(math.sqrt(float(hidden)))
    ple_scale = f32(1.0) / embed_scale

    table = weights["language_model.embed_tokens.weight"]
    states = np.empty((seq, hidden), dtype=np.float32)
    for t, token in enumerate(tokens):
        states[t] = table[token] * embed_scale

    ple = gemm(states, weights["language_model.ple.per_layer_model_projection.weight"])
    ple = ple * ple_scale
    ple = ple.reshape(seq * n_layers, ple_dim)
    ple = rms_norm(ple, weights["language_model.ple.per_layer_projection_norm.weight"], eps)
    ple = ple.reshape(seq, n_layers, ple_dim)

    for layer in range(n_layers):
        spec = layer_spec(text, layer)
        prefix = f"language_model.layers.{layer}."
        head_dim = spec["head_dim"]
        n_kv = spec["n_kv"]
        residual = states.copy()
        normed = rms_norm(states, weights[prefix + "input_layernorm.weight"], eps)
        q = gemm(normed, weights[prefix + "self_attn.q_proj.weight"]).reshape(seq, n_heads, head_dim)
        k = gemm(normed, weights[prefix + "self_attn.k_proj.weight"]).reshape(seq, n_kv, head_dim)
        v = gemm(normed, weights[prefix + "self_attn.v_proj.weight"]).reshape(seq, n_kv, head_dim)
        q = rms_norm(q.reshape(seq * n_heads, head_dim), weights[prefix + "self_attn.q_norm.weight"], eps)
        k = rms_norm(k.reshape(seq * n_kv, head_dim), weights[prefix + "self_attn.k_norm.weight"], eps)
        v = rms_norm(v.reshape(seq * n_kv, head_dim), None, eps)
        q = q.reshape(seq, n_heads, head_dim)
        k = k.reshape(seq, n_kv, head_dim)
        v = v.reshape(seq, n_kv, head_dim)
        freq = inv_freq(spec["theta"], head_dim)
        apply_rope(q, freq)
        apply_rope(k, freq)
        attn = attention(q, k, v, not spec["full"], window).reshape(seq, n_heads * head_dim)
        mixed = gemm(attn, weights[prefix + "self_attn.o_proj.weight"])
        mixed = rms_norm(mixed, weights[prefix + "post_attention_layernorm.weight"], eps)
        states = residual + mixed

        residual = states.copy()
        normed = rms_norm(states, weights[prefix + "pre_feedforward_layernorm.weight"], eps)
        gate = gemm(normed, weights[prefix + "mlp.gate_proj.weight"])
        up = gemm(normed, weights[prefix + "mlp.up_proj.weight"])
        gate = gelu_tanh(gate) * up
        mixed = gemm(gate, weights[prefix + "mlp.down_proj.weight"])
        mixed = rms_norm(mixed, weights[prefix + "post_feedforward_layernorm.weight"], eps)
        states = residual + mixed

        residual = states.copy()
        ple_gate = gemm(states, weights[prefix + "ple_block.per_layer_input_gate.weight"])
        ple_gate = gelu_tanh(ple_gate) * ple[:, layer, :]
        mixed = gemm(ple_gate, weights[prefix + "ple_block.per_layer_projection.weight"])
        mixed = rms_norm(mixed, weights[prefix + "ple_block.post_per_layer_input_norm.weight"], eps)
        scalar = f32(weights[prefix + "layer_scalar"][0])
        states = (residual + mixed) * scalar

    states = rms_norm(states, weights["language_model.norm.weight"], eps)
    projected = gemm(states, weights["language_model.embedding_projection.weight"])
    pooled = np.zeros(embed_dim, dtype=np.float32)
    for row in projected:
        for d in range(embed_dim):
            pooled[d] += row[d]
    pooled /= f32(seq)
    return l2_normalize(pooled)


def _tensor_names(text: dict) -> list[tuple[str, tuple[int, ...]]]:
    hidden = text["hidden_size"]
    inter = text["intermediate_size"]
    ple = text["hidden_size_per_layer_input"]
    embed = text["embedding_dim"]
    vocab = text["vocab_size"]
    n_layers = text["num_hidden_layers"]
    n_heads = text["num_attention_heads"]
    names = [
        ("language_model.embed_tokens.weight", (vocab, hidden)),
        ("language_model.norm.weight", (hidden,)),
        ("language_model.embedding_projection.weight", (embed, hidden)),
        ("language_model.ple.per_layer_model_projection.weight", (n_layers * ple, hidden)),
        ("language_model.ple.per_layer_projection_norm.weight", (ple,)),
    ]
    for i in range(n_layers):
        spec = layer_spec(text, i)
        q_dim = n_heads * spec["head_dim"]
        kv_dim = spec["n_kv"] * spec["head_dim"]
        prefix = f"language_model.layers.{i}."
        names.extend(
            [
                (prefix + "input_layernorm.weight", (hidden,)),
                (prefix + "post_attention_layernorm.weight", (hidden,)),
                (prefix + "pre_feedforward_layernorm.weight", (hidden,)),
                (prefix + "post_feedforward_layernorm.weight", (hidden,)),
                (prefix + "self_attn.q_proj.weight", (q_dim, hidden)),
                (prefix + "self_attn.k_proj.weight", (kv_dim, hidden)),
                (prefix + "self_attn.v_proj.weight", (kv_dim, hidden)),
                (prefix + "self_attn.o_proj.weight", (hidden, q_dim)),
                (prefix + "self_attn.q_norm.weight", (spec["head_dim"],)),
                (prefix + "self_attn.k_norm.weight", (spec["head_dim"],)),
                (prefix + "mlp.gate_proj.weight", (inter, hidden)),
                (prefix + "mlp.up_proj.weight", (inter, hidden)),
                (prefix + "mlp.down_proj.weight", (hidden, inter)),
                (prefix + "ple_block.per_layer_input_gate.weight", (ple, hidden)),
                (prefix + "ple_block.per_layer_projection.weight", (hidden, ple)),
                (prefix + "ple_block.post_per_layer_input_norm.weight", (hidden,)),
                (prefix + "layer_scalar", (1,)),
            ]
        )
    return names


def make_tiny_weights(text: dict, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    weights = {}
    for name, shape in _tensor_names(text):
        if name.endswith("layer_scalar") or name.endswith("norm.weight") or name.endswith("_norm.weight"):
            values = 0.75 + 0.5 * rng.random(shape)
        else:
            values = rng.normal(0.0, 0.15, size=shape)
        rounded = bf16_to_f32(f32_to_bf16(values.astype(np.float32)))
        weights[name] = rounded
    return weights


def write_safetensors(path: Path, weights: dict[str, np.ndarray]) -> None:
    blobs = []
    header = {}
    cursor = 0
    for name, value in weights.items():
        raw = f32_to_bf16(value).tobytes()
        header[name] = {
            "dtype": "BF16",
            "shape": list(value.shape),
            "data_offsets": [cursor, cursor + len(raw)],
        }
        blobs.append(raw)
        cursor += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(encoded) % 8)) % 8
    encoded += b" " * pad
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


def write_fixture(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    text = TINY_CONFIG["text_config"]
    weights = make_tiny_weights(text)
    outputs = [forward(text, weights, tokens) for tokens in TINY_SEQUENCES]
    (directory / "config.json").write_text(json.dumps(TINY_CONFIG, indent=2) + "\n")
    write_safetensors(directory / "model.safetensors", weights)
    token_lines = [str(len(TINY_SEQUENCES))]
    for tokens in TINY_SEQUENCES:
        token_lines.append(str(len(tokens)) + " " + " ".join(str(token) for token in tokens))
    (directory / "tokens.txt").write_text("\n".join(token_lines) + "\n")
    blob = b"".join(vector.astype("<f4").tobytes() for vector in outputs)
    (directory / "expected.f32").write_bytes(blob)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    write_fixture(args.fixture)


if __name__ == "__main__":
    main()
