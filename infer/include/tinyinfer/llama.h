#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "tinyinfer/backend.h"

namespace tinyinfer {

using TokenId = uint32_t;

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

struct NamedTensor {
    std::string name;
    Tensor* tensor = nullptr;
};

struct LlamaLayerWeights {
    Tensor attn_norm;
    Tensor q_proj;
    Tensor k_proj;
    Tensor v_proj;
    Tensor o_proj;
    Tensor ffn_norm;
    Tensor gate_proj;
    Tensor up_proj;
    Tensor down_proj;
};

struct LlamaModel {
    LlamaConfig config;
    Tensor token_embedding;
    Tensor final_norm;
    Tensor lm_head;
    std::vector<LlamaLayerWeights> layers;

    std::vector<NamedTensor> parameters();
};

struct KVCache {
    Tensor keys;
    Tensor values;
    uint32_t seq_len = 0;
    uint32_t max_seq_len = 0;
};

struct GenerateConfig {
    uint32_t max_new_tokens = 0;
    TokenId eos_token_id = 2;
    bool stop_on_eos = true;
};

class LlamaRunner {
public:
    LlamaRunner(Backend& backend, LlamaModel& model);

    Status init_kv_cache(KVCache& cache, uint32_t max_seq_len);

    Status prefill(std::span<const TokenId> prompt, KVCache& cache, Tensor& logits);
    Status decode_one(TokenId token, KVCache& cache, Tensor& logits, TokenId& next_token);
    Status generate(
        std::span<const TokenId> prompt,
        std::span<TokenId> output,
        const GenerateConfig& config,
        uint32_t& output_count);

private:
    Status check_ready() const;
    Status run_layers(uint32_t start_pos, uint32_t seq_len, KVCache& cache, Tensor& logits);

    Backend& backend_;
    LlamaModel& model_;
    Stream stream_;
    Status stream_status_;
};

}  // namespace tinyinfer
