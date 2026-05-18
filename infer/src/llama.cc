#include "tinyinfer/llama.h"

#include <array>
#include <string>
#include <utility>

namespace tinyinfer {
namespace {

Status validate_prompt(
    std::span<const TokenId> prompt,
    const LlamaConfig& config,
    const KVCache& cache,
    uint32_t start_pos) {
    if (prompt.empty()) {
        return Status::invalid_argument_status("prompt must not be empty");
    }
    if (start_pos > cache.max_seq_len) {
        return Status::invalid_argument_status("start position exceeds KV cache capacity");
    }
    if (start_pos > config.max_seq_len) {
        return Status::invalid_argument_status("start position exceeds model max sequence length");
    }
    if (prompt.size() > cache.max_seq_len - start_pos) {
        return Status::invalid_argument_status("sequence exceeds KV cache capacity");
    }
    if (prompt.size() > config.max_seq_len - start_pos) {
        return Status::invalid_argument_status("sequence exceeds model max sequence length");
    }
    return Status::success();
}

}  // namespace

uint32_t LlamaConfig::head_dim() const {
    return n_heads == 0 ? 0 : hidden_size / n_heads;
}

uint32_t LlamaConfig::kv_group_size() const {
    return n_kv_heads == 0 ? 0 : n_heads / n_kv_heads;
}

Status LlamaConfig::validate() const {
    if (n_layers == 0) {
        return Status::invalid_config_status("n_layers must be non-zero");
    }
    if (hidden_size == 0) {
        return Status::invalid_config_status("hidden_size must be non-zero");
    }
    if (intermediate_size == 0) {
        return Status::invalid_config_status("intermediate_size must be non-zero");
    }
    if (n_heads == 0) {
        return Status::invalid_config_status("n_heads must be non-zero");
    }
    if (n_kv_heads == 0) {
        return Status::invalid_config_status("n_kv_heads must be non-zero");
    }
    if (vocab_size == 0) {
        return Status::invalid_config_status("vocab_size must be non-zero");
    }
    if (max_seq_len == 0) {
        return Status::invalid_config_status("max_seq_len must be non-zero");
    }
    if (rms_eps <= 0.0f) {
        return Status::invalid_config_status("rms_eps must be positive");
    }
    if (rope_theta <= 0.0f) {
        return Status::invalid_config_status("rope_theta must be positive");
    }
    if (hidden_size % n_heads != 0) {
        return Status::invalid_config_status("hidden_size must be divisible by n_heads");
    }
    if (n_heads % n_kv_heads != 0) {
        return Status::invalid_config_status("n_heads must be divisible by n_kv_heads");
    }
    if ((head_dim() % 2) != 0) {
        return Status::invalid_config_status("head_dim must be even for RoPE");
    }
    return Status::success();
}

LlamaConfig LlamaConfig::tinyllama_1_1b() {
    return {
        22,
        2048,
        5632,
        32,
        4,
        32000,
        2048,
        1e-5f,
        10000.0f,
    };
}

LlamaConfig LlamaConfig::demo() {
    return {
        2,
        128,
        256,
        4,
        2,
        256,
        128,
        1e-5f,
        10000.0f,
    };
}

std::vector<NamedTensor> LlamaModel::parameters() {
    std::vector<NamedTensor> params;
    params.reserve(3 + layers.size() * 9);

    params.push_back({"token_embedding", &token_embedding});
    params.push_back({"final_norm", &final_norm});
    params.push_back({"lm_head", &lm_head});

    for (size_t i = 0; i < layers.size(); ++i) {
        LlamaLayerWeights& layer = layers[i];
        const std::string prefix = "layers." + std::to_string(i) + ".";

        params.push_back({prefix + "attn_norm", &layer.attn_norm});
        params.push_back({prefix + "q_proj", &layer.q_proj});
        params.push_back({prefix + "k_proj", &layer.k_proj});
        params.push_back({prefix + "v_proj", &layer.v_proj});
        params.push_back({prefix + "o_proj", &layer.o_proj});
        params.push_back({prefix + "ffn_norm", &layer.ffn_norm});
        params.push_back({prefix + "gate_proj", &layer.gate_proj});
        params.push_back({prefix + "up_proj", &layer.up_proj});
        params.push_back({prefix + "down_proj", &layer.down_proj});
    }

    return params;
}

LlamaRunner::LlamaRunner(Backend& backend, LlamaModel& model)
    : backend_(backend),
      model_(model),
      stream_status_(Status::success()) {
    Result<Stream> stream = backend_.new_stream();
    stream_status_ = stream.status;
    if (stream.status) {
        stream_ = stream.value;
    }
}

Status LlamaRunner::check_ready() const {
    if (!stream_status_) {
        return stream_status_;
    }
    Status status = model_.config.validate();
    if (!status) {
        return status;
    }
    if (model_.layers.size() != model_.config.n_layers) {
        return Status::invalid_config_status("model layer count does not match config");
    }
    return Status::success();
}

Status LlamaRunner::init_kv_cache(KVCache& cache, uint32_t max_seq_len) {
    Status status = check_ready();
    if (!status) {
        return status;
    }
    if (max_seq_len == 0 || max_seq_len > model_.config.max_seq_len) {
        return Status::invalid_argument_status("invalid KV cache sequence length");
    }

    const Shape shape = make_shape({
        static_cast<int64_t>(model_.config.n_layers),
        static_cast<int64_t>(model_.config.n_kv_heads),
        static_cast<int64_t>(max_seq_len),
        static_cast<int64_t>(model_.config.head_dim()),
    });

    Result<Tensor> keys = backend_.empty(shape, DType::f32);
    if (!keys.status) {
        return keys.status;
    }
    Result<Tensor> values = backend_.empty(shape, DType::f32);
    if (!values.status) {
        return values.status;
    }

    cache.keys = std::move(keys.value);
    cache.values = std::move(values.value);
    cache.seq_len = 0;
    cache.max_seq_len = max_seq_len;
    return Status::success();
}

Status LlamaRunner::prefill(std::span<const TokenId> prompt, KVCache& cache, Tensor& logits) {
    Status status = check_ready();
    if (!status) {
        return status;
    }
    status = validate_prompt(prompt, model_.config, cache, 0);
    if (!status) {
        return status;
    }
    if (cache.seq_len != 0) {
        return Status::invalid_argument_status("prefill requires an empty KV cache");
    }

    status = run_layers(0, static_cast<uint32_t>(prompt.size()), cache, logits);
    if (!status) {
        return status;
    }

    cache.seq_len = static_cast<uint32_t>(prompt.size());
    return Status::success();
}

Status LlamaRunner::decode_one(
    TokenId token,
    KVCache& cache,
    Tensor& logits,
    TokenId& next_token) {
    (void)token;

    Status status = check_ready();
    if (!status) {
        return status;
    }
    const std::array<TokenId, 1> one_token = {token};
    status = validate_prompt(one_token, model_.config, cache, cache.seq_len);
    if (!status) {
        return status;
    }

    status = run_layers(cache.seq_len, 1, cache, logits);
    if (!status) {
        return status;
    }

    status = backend_.argmax(next_token, logits, stream_);
    if (!status) {
        return status;
    }

    ++cache.seq_len;
    return Status::success();
}

Status LlamaRunner::generate(
    std::span<const TokenId> prompt,
    std::span<TokenId> output,
    const GenerateConfig& config,
    uint32_t& output_count) {
    output_count = 0;
    Status status = check_ready();
    if (!status) {
        return status;
    }
    if (config.max_new_tokens == 0) {
        return Status::invalid_argument_status("max_new_tokens must be non-zero");
    }
    if (output.size() < prompt.size() + config.max_new_tokens) {
        return Status::invalid_argument_status("output buffer is too small");
    }

    KVCache cache;
    status = init_kv_cache(cache, model_.config.max_seq_len);
    if (!status) {
        return status;
    }

    Tensor logits;
    status = prefill(prompt, cache, logits);
    if (!status) {
        return status;
    }

    for (size_t i = 0; i < prompt.size(); ++i) {
        output[i] = prompt[i];
    }
    output_count = static_cast<uint32_t>(prompt.size());

    TokenId token = prompt.back();
    for (uint32_t i = 0; i < config.max_new_tokens; ++i) {
        TokenId next = 0;
        status = decode_one(token, cache, logits, next);
        if (!status) {
            return status;
        }

        output[output_count++] = next;
        token = next;
        if (config.stop_on_eos && next == config.eos_token_id) {
            break;
        }
    }

    return Status::success();
}

Status LlamaRunner::run_layers(
    uint32_t start_pos,
    uint32_t seq_len,
    KVCache& cache,
    Tensor& logits) {
    (void)seq_len;

    Tensor hidden;
    Tensor normed;
    Tensor q;
    Tensor k;
    Tensor v;
    Tensor attn_out;
    Tensor gate;
    Tensor up;
    Tensor swiglu;
    Tensor ffn_out;

    for (uint32_t i = 0; i < model_.config.n_layers; ++i) {
        LlamaLayerWeights& layer = model_.layers[i];

        Status status = backend_.rms_norm_out(normed, hidden, layer.attn_norm, model_.config.rms_eps, stream_);
        if (!status) {
            return status;
        }
        status = backend_.matmul_out(q, normed, layer.q_proj, stream_);
        if (!status) {
            return status;
        }
        status = backend_.matmul_out(k, normed, layer.k_proj, stream_);
        if (!status) {
            return status;
        }
        status = backend_.matmul_out(v, normed, layer.v_proj, stream_);
        if (!status) {
            return status;
        }
        status = backend_.rope_inplace(q, k, start_pos, stream_);
        if (!status) {
            return status;
        }
        status = backend_.attention_out(
            attn_out,
            q,
            cache.keys,
            cache.values,
            start_pos + seq_len,
            stream_);
        if (!status) {
            return status;
        }
        status = backend_.matmul_out(hidden, attn_out, layer.o_proj, stream_);
        if (!status) {
            return status;
        }
        status = backend_.rms_norm_out(normed, hidden, layer.ffn_norm, model_.config.rms_eps, stream_);
        if (!status) {
            return status;
        }
        status = backend_.matmul_out(gate, normed, layer.gate_proj, stream_);
        if (!status) {
            return status;
        }
        status = backend_.matmul_out(up, normed, layer.up_proj, stream_);
        if (!status) {
            return status;
        }
        status = backend_.swiglu_out(swiglu, gate, up, stream_);
        if (!status) {
            return status;
        }
        status = backend_.matmul_out(ffn_out, swiglu, layer.down_proj, stream_);
        if (!status) {
            return status;
        }

        hidden = std::move(ffn_out);
    }

    Status status = backend_.rms_norm_out(hidden, hidden, model_.final_norm, model_.config.rms_eps, stream_);
    if (!status) {
        return status;
    }
    return backend_.matmul_out(logits, hidden, model_.lm_head, stream_);
}

}  // namespace tinyinfer
