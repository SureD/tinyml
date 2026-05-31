#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "tinyinfer/backend.h"

namespace tinyinfer {

using TokenId = uint32_t;

class LlamaInferEngine;
struct HfModelFiles;

struct LlamaConfig {
    uint32_t n_layers = 0;
    uint32_t hidden_size = 0;
    uint32_t intermediate_size = 0;
    uint32_t n_heads = 0;
    uint32_t n_kv_heads = 0;
    uint32_t vocab_size = 0;
    uint32_t max_seq_len = 0;
    float rms_eps = 0.0f;
    float rope_theta = 0.0f;

    uint32_t head_dim() const;
    uint32_t kv_group_size() const;
    Status validate() const;

    static LlamaConfig tinyllama_1_1b();
    static LlamaConfig demo();
};

struct GenerateConfig {
    uint32_t max_new_tokens = 0;
    TokenId eos_token_id = 2;
    bool stop_on_eos = true;
};

class LlamaInferEngine {
public:
    LlamaInferEngine(LlamaInferEngine&& other) noexcept;
    LlamaInferEngine& operator=(LlamaInferEngine&& other) noexcept;
    LlamaInferEngine(const LlamaInferEngine&) = delete;
    LlamaInferEngine& operator=(const LlamaInferEngine&) = delete;

    static Result<LlamaInferEngine> create(
        Backend& backend,
        const LlamaConfig& config,
        uint32_t max_seq_len);

    void reset();
    Status prefill(std::span<const TokenId> prompt, TokenId& next_token);
    Status decode_one(TokenId token, TokenId& next_token);
    Status generate(
        std::span<const TokenId> prompt,
        std::span<TokenId> output,
        const GenerateConfig& config,
        uint32_t& output_count);

    const LlamaConfig& config() const;
    uint32_t seq_len() const;
    uint32_t max_seq_len() const;

private:
    friend Status load_llama_safetensors(
        LlamaInferEngine& engine,
        const char* path);
    friend Result<LlamaInferEngine> load_llama_from_hf_dir(
        Backend& backend,
        const char* model_dir,
        uint32_t max_seq_len);
    friend Result<LlamaInferEngine> load_llama_from_hf_files(
        Backend& backend,
        const HfModelFiles& files,
        uint32_t max_seq_len);

    struct LayerWeights {
        TensorView attn_norm;
        TensorView q_proj;
        TensorView k_proj;
        TensorView v_proj;
        TensorView o_proj;
        TensorView ffn_norm;
        TensorView gate_proj;
        TensorView up_proj;
        TensorView down_proj;
    };

    struct Model {
        TensorView token_embedding;
        TensorView final_norm;
        TensorView lm_head;
        std::vector<LayerWeights> layers;
    };

    struct KVCache {
        TensorView keys;
        TensorView values;
        uint32_t seq_len = 0;
        uint32_t max_seq_len = 0;
    };

    LlamaInferEngine() = default;
    LlamaInferEngine(Backend& backend, const LlamaConfig& config, uint32_t max_seq_len);

    Status init();
    Status bind_model();
    Status init_kv_cache();
    void rebind_views_after_move(const LlamaInferEngine& source);
    void rebind_view_after_move(TensorView& view, const LlamaInferEngine& source);
    TensorView workspace_tensor(const Shape& shape);
    void run_layers(std::span<const TokenId> tokens, uint32_t start_pos, TensorView& logits);

    Backend* backend_ = nullptr;
    LlamaConfig config_;
    uint32_t max_seq_len_ = 0;
    MemoryArena weights_;
    MemoryArena kv_cache_arena_;
    MemoryArena workspace_;
    Model model_;
    KVCache cache_;
    TensorView logits_;
};

}  // namespace tinyinfer
