#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "tinyinfer/backend.h"

namespace tinyinfer {

// One layer of the EmbeddingGemma 2 text tower.
// Local layers and global layers do not share a head dim. The checkpoint
// stores that split in `per_layer_config`; do not derive it from hidden_size.
struct EmbeddingGemma2Layer {
    uint32_t head_dim = 0;
    uint32_t n_kv_heads = 0;
    bool full_attention = false;
    float rope_theta = 0.0f;
};

struct EmbeddingGemma2Config {
    uint32_t n_layers = 0;
    uint32_t hidden_size = 0;
    uint32_t intermediate_size = 0;
    uint32_t n_heads = 0;
    uint32_t vocab_size = 0;
    uint32_t embedding_dim = 0;
    uint32_t ple_dim = 0;
    // Inclusive radius: a sliding layer attends where abs(q - k) <= sliding_window.
    uint32_t sliding_window = 0;
    uint32_t max_position_embeddings = 0;
    float rms_eps = 0.0f;
    std::vector<EmbeddingGemma2Layer> layers;

    uint32_t q_dim(uint32_t layer) const;
    uint32_t kv_dim(uint32_t layer) const;
    uint32_t max_q_dim() const;
    uint32_t max_kv_dim() const;
    Status validate() const;
};

// Text embedder. One call takes token ids and writes one L2-normalized vector.
// Batch 1. There is no KV cache: each layer's K and V live in the workspace
// and are overwritten by the next layer.
//
// Metal is not implemented. embed() on a non-CPU backend returns unimplemented.
class EmbeddingGemma2Engine {
public:
    EmbeddingGemma2Engine(EmbeddingGemma2Engine&& other) noexcept;
    EmbeddingGemma2Engine& operator=(EmbeddingGemma2Engine&& other) noexcept;
    EmbeddingGemma2Engine(const EmbeddingGemma2Engine&) = delete;
    EmbeddingGemma2Engine& operator=(const EmbeddingGemma2Engine&) = delete;

    static Result<EmbeddingGemma2Engine> create(
        Backend& backend,
        const EmbeddingGemma2Config& config,
        uint32_t max_seq_len);

    // `embedding` length is `config.embedding_dim`, or a Matryoshka width of
    // 128, 256, or 512. A shorter width keeps the leading values and
    // L2-normalizes again. Every token in `tokens` is real; the caller does
    // not pass bucket padding.
    Status embed(
        std::span<const uint32_t> tokens,
        std::span<float> embedding);

    const EmbeddingGemma2Config& config() const;
    uint32_t max_seq_len() const;
    size_t weight_arena_bytes() const;
    size_t workspace_arena_bytes() const;
    bool weights_loaded() const;

private:
    friend Result<EmbeddingGemma2Config> load_embeddinggemma2_config_json(const char* path);
    friend Status load_embeddinggemma2_safetensors(
        EmbeddingGemma2Engine& engine,
        const char* path);
    friend Result<EmbeddingGemma2Engine> load_embeddinggemma2_from_hf_dir(
        Backend& backend,
        const char* model_dir,
        uint32_t max_seq_len);

    struct LayerWeights {
        TensorView input_layernorm;
        TensorView post_attention_layernorm;
        TensorView pre_feedforward_layernorm;
        TensorView post_feedforward_layernorm;
        TensorView q_proj;
        TensorView k_proj;
        TensorView v_proj;
        TensorView o_proj;
        TensorView q_norm;
        TensorView k_norm;
        TensorView gate_proj;
        TensorView up_proj;
        TensorView down_proj;
        TensorView ple_gate;
        TensorView ple_proj;
        TensorView ple_norm;
        TensorView layer_scalar;
    };

    struct Model {
        TensorView embed_tokens;
        TensorView final_norm;
        TensorView embedding_projection;
        TensorView ple_projection;
        TensorView ple_norm;
        std::vector<LayerWeights> layers;
    };

    struct Workspace {
        TensorView hidden;
        TensorView residual;
        TensorView normed;
        TensorView q;
        TensorView k;
        TensorView v;
        TensorView attn;
        TensorView gate;
        TensorView up;
        TensorView ple_h;
        TensorView ple;
        TensorView projected;
        TensorView pooled;
    };

    EmbeddingGemma2Engine() = default;
    EmbeddingGemma2Engine(
        Backend& backend,
        const EmbeddingGemma2Config& config,
        uint32_t max_seq_len);

    Status init();
    Status bind_weights();
    Status bind_workspace();
    void rebind_view(TensorView& view, const EmbeddingGemma2Engine& source);
    void rebind_views_after_move(const EmbeddingGemma2Engine& source);
    Status forward(
        std::span<const uint32_t> tokens,
        std::span<float> embedding);

    Backend* backend_ = nullptr;
    EmbeddingGemma2Config config_;
    uint32_t max_seq_len_ = 0;
    bool weights_loaded_ = false;
    MemoryArena weights_;
    MemoryArena workspace_;
    Model model_;
    Workspace work_;
    std::vector<std::vector<float>> rope_inv_freq_;
};

Result<EmbeddingGemma2Config> load_embeddinggemma2_config_json(const char* path);

Status load_embeddinggemma2_safetensors(
    EmbeddingGemma2Engine& engine,
    const char* path);

// `model_dir` contains config.json and model.safetensors, as in a
// Hugging Face checkout. Vision and audio tensors are skipped.
Result<EmbeddingGemma2Engine> load_embeddinggemma2_from_hf_dir(
    Backend& backend,
    const char* model_dir,
    uint32_t max_seq_len);

}  // namespace tinyinfer
