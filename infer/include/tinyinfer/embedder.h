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

// Weight packing for the text matrices. f32 is the loaded checkpoint.
// int8 and int4 are per output row, with a scale and a zero point, because
// the checkpoint does not say the integers are symmetric. This is TinyInfer's
// layout, not a LiteRT QAT file.
enum class EmbeddingGemma2Quant {
    f32,
    int8,
    int4,
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

    // Repacks every 2-D weight from the loaded float32 values.
    // One-dimensional norms and layer scalars stay float32.
    // The unpacked float matrices are not kept.
    Status quantize_weights(EmbeddingGemma2Quant quant);

    const EmbeddingGemma2Config& config() const;
    uint32_t max_seq_len() const;
    size_t weight_arena_bytes() const;
    size_t workspace_arena_bytes() const;
    bool weights_loaded() const;
    EmbeddingGemma2Quant weight_quant() const;

private:
    friend Result<EmbeddingGemma2Config> load_embeddinggemma2_config_json(const char* path);
    friend Status load_embeddinggemma2_safetensors(
        EmbeddingGemma2Engine& engine,
        const char* path);
    friend Result<EmbeddingGemma2Engine> load_embeddinggemma2_from_hf_dir(
        Backend& backend,
        const char* model_dir,
        uint32_t max_seq_len);

    // `weight` is float32 [rows, cols], or packed int8 / int4 after quantize.
    // `scale` and `zero` are one per output row. int4 packs two values per byte,
    // low nibble first, sign-extended on read.
    struct Matrix {
        TensorView weight;
        TensorView scale;
        TensorView zero;
        int64_t rows = 0;
        int64_t cols = 0;
    };

    struct LayerWeights {
        TensorView input_layernorm;
        TensorView post_attention_layernorm;
        TensorView pre_feedforward_layernorm;
        TensorView post_feedforward_layernorm;
        Matrix q_proj;
        Matrix k_proj;
        Matrix v_proj;
        Matrix o_proj;
        TensorView q_norm;
        TensorView k_norm;
        Matrix gate_proj;
        Matrix up_proj;
        Matrix down_proj;
        Matrix ple_gate;
        Matrix ple_proj;
        TensorView ple_norm;
        TensorView layer_scalar;
    };

    struct Model {
        Matrix embed_tokens;
        TensorView final_norm;
        Matrix embedding_projection;
        Matrix ple_projection;
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
        // One softmax-score row, length max_seq_len. Reused by every layer.
        TensorView attn_score;
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
    void rebind_matrix(Matrix& matrix, const EmbeddingGemma2Engine& source);
    void rebind_views_after_move(const EmbeddingGemma2Engine& source);
    Status forward(
        std::span<const uint32_t> tokens,
        std::span<float> embedding);

    Backend* backend_ = nullptr;
    EmbeddingGemma2Config config_;
    uint32_t max_seq_len_ = 0;
    bool weights_loaded_ = false;
    EmbeddingGemma2Quant quant_ = EmbeddingGemma2Quant::f32;
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
