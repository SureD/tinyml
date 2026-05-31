#include "tinyinfer/llama.h"

#include <array>
#include <cstddef>
#include <utility>

namespace tinyinfer {
namespace {

struct MemoryPlan {
    size_t weights_bytes = 0;
    size_t kv_cache_bytes = 0;
    size_t workspace_bytes = 0;
};

size_t f32_bytes(size_t elems) {
    return elems * dtype_size(DType::f32);
}

MemoryPlan build_memory_plan(const LlamaConfig& config, uint32_t max_seq_len) {
    MemoryPlan plan;
    if (!config.validate()) {
        return plan;
    }
    if (max_seq_len == 0 || max_seq_len > config.max_seq_len) {
        return plan;
    }

    const size_t hidden = config.hidden_size;
    const size_t inter = config.intermediate_size;
    const size_t vocab = config.vocab_size;
    const size_t head_dim = config.head_dim();
    const size_t kv_dim = config.n_kv_heads * head_dim;

    const size_t global_weight_elems =
        vocab * hidden +  // token_embedding
        hidden +          // final_norm
        vocab * hidden;   // lm_head

    const size_t layer_elems =
        hidden +          // attn_norm
        hidden * hidden + // q_proj
        kv_dim * hidden + // k_proj
        kv_dim * hidden + // v_proj
        hidden * hidden + // o_proj
        hidden +          // ffn_norm
        inter * hidden +  // gate_proj
        inter * hidden +  // up_proj
        hidden * inter;   // down_proj

    constexpr size_t kTensorAlignmentSlack = 64;
    constexpr size_t kGlobalWeightTensors = 3;
    constexpr size_t kLayerWeightTensors = 9;
    const size_t layer_count = config.n_layers;

    const size_t weight_elems = global_weight_elems + layer_count * layer_elems;
    const size_t weight_tensor_count =
        kGlobalWeightTensors + layer_count * kLayerWeightTensors;
    const size_t weight_alignment_slack =
        weight_tensor_count * kTensorAlignmentSlack;

    plan.weights_bytes = f32_bytes(weight_elems) + weight_alignment_slack;

    const size_t kv_elems =
        2ULL *
        config.n_layers *
        config.n_kv_heads *
        max_seq_len *
        head_dim;
    plan.kv_cache_bytes = f32_bytes(kv_elems) + 2 * kTensorAlignmentSlack;

    const size_t workspace_elems =
        7ULL * max_seq_len * hidden +
        3ULL * max_seq_len * inter +
        max_seq_len * vocab;
    plan.workspace_bytes = f32_bytes(workspace_elems) + 16 * kTensorAlignmentSlack;

    return plan;
}

Status validate_prompt(
    std::span<const TokenId> prompt,
    const LlamaConfig& config,
    uint32_t cache_max_seq_len,
    uint32_t start_pos) {
    if (prompt.empty()) {
        return Status::invalid_argument_status("prompt must not be empty");
    }
    if (start_pos > cache_max_seq_len) {
        return Status::invalid_argument_status("start position exceeds KV cache capacity");
    }
    if (start_pos > config.max_seq_len) {
        return Status::invalid_argument_status("start position exceeds model max sequence length");
    }
    if (prompt.size() > cache_max_seq_len - start_pos) {
        return Status::invalid_argument_status("sequence exceeds KV cache capacity");
    }
    if (prompt.size() > config.max_seq_len - start_pos) {
        return Status::invalid_argument_status("sequence exceeds model max sequence length");
    }
    for (TokenId token : prompt) {
        if (token >= config.vocab_size) {
            return Status::invalid_argument_status("token id exceeds vocab size");
        }
    }
    return Status::success();
}

Status alloc_view(MemoryArena& arena, const Shape& shape, TensorView& out) {
    Result<TensorView> view = arena.alloc(shape, DType::f32);
    if (!view.status) {
        return view.status;
    }
    out = view.value;
    return Status::success();
}

TensorView layer_cache_view(const TensorView& cache, uint32_t layer) {
    TensorView view = cache;
    view.byte_offset +=
        static_cast<size_t>(layer) *
        static_cast<size_t>(cache.strides.stride(0)) *
        cache.item_size();
    view.shape = make_shape({cache.dim(1), cache.dim(2), cache.dim(3)});
    view.strides.ndim = 3;
    view.strides.values[0] = cache.strides.stride(1);
    view.strides.values[1] = cache.strides.stride(2);
    view.strides.values[2] = cache.strides.stride(3);
    return view;
}

TensorView last_token_view(const TensorView& values, uint32_t token_index) {
    TensorView view = values;
    view.byte_offset +=
        static_cast<size_t>(token_index) *
        static_cast<size_t>(values.strides.stride(0)) *
        values.item_size();
    view.shape = make_shape({1, values.dim(1)});
    view.strides.ndim = 2;
    view.strides.values[0] = values.strides.stride(0);
    view.strides.values[1] = values.strides.stride(1);
    return view;
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

LlamaInferEngine::LlamaInferEngine(
    Backend& backend,
    const LlamaConfig& config,
    uint32_t max_seq_len)
    : backend_(&backend),
      config_(config),
      max_seq_len_(max_seq_len) {}

LlamaInferEngine::LlamaInferEngine(LlamaInferEngine&& other) noexcept
    : backend_(other.backend_),
      config_(other.config_),
      max_seq_len_(other.max_seq_len_),
      weights_(std::move(other.weights_)),
      kv_cache_arena_(std::move(other.kv_cache_arena_)),
      workspace_(std::move(other.workspace_)),
      model_(std::move(other.model_)),
      cache_(other.cache_),
      logits_(other.logits_) {
    rebind_views_after_move(other);

    other.backend_ = nullptr;
    other.max_seq_len_ = 0;
    other.cache_ = {};
    other.logits_ = {};
}

LlamaInferEngine& LlamaInferEngine::operator=(LlamaInferEngine&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    backend_ = other.backend_;
    config_ = other.config_;
    max_seq_len_ = other.max_seq_len_;
    weights_ = std::move(other.weights_);
    kv_cache_arena_ = std::move(other.kv_cache_arena_);
    workspace_ = std::move(other.workspace_);
    model_ = std::move(other.model_);
    cache_ = other.cache_;
    logits_ = other.logits_;
    rebind_views_after_move(other);

    other.backend_ = nullptr;
    other.max_seq_len_ = 0;
    other.cache_ = {};
    other.logits_ = {};
    return *this;
}

Result<LlamaInferEngine> LlamaInferEngine::create(
    Backend& backend,
    const LlamaConfig& config,
    uint32_t max_seq_len) {
    LlamaInferEngine engine(backend, config, max_seq_len);
    Status status = engine.init();
    if (!status) {
        return {status, LlamaInferEngine()};
    }
    return {Status::success(), std::move(engine)};
}

Status LlamaInferEngine::init() {
    Status status = config_.validate();
    if (!status) {
        return status;
    }
    if (max_seq_len_ == 0 || max_seq_len_ > config_.max_seq_len) {
        return Status::invalid_argument_status("invalid max sequence length");
    }

    const MemoryPlan plan = build_memory_plan(config_, max_seq_len_);

    status = backend_->alloc_arena(weights_, plan.weights_bytes, MemoryKind::weights);
    if (!status) {
        return status;
    }
    status = backend_->alloc_arena(kv_cache_arena_, plan.kv_cache_bytes, MemoryKind::kv_cache);
    if (!status) {
        return status;
    }
    status = backend_->alloc_arena(workspace_, plan.workspace_bytes, MemoryKind::workspace);
    if (!status) {
        return status;
    }

    status = bind_model();
    if (!status) {
        return status;
    }
    return init_kv_cache();
}

Status LlamaInferEngine::bind_model() {
    weights_.reset();
    model_.layers.resize(config_.n_layers);

    Status status = alloc_view(
        weights_,
        make_shape({
            static_cast<int64_t>(config_.vocab_size),
            static_cast<int64_t>(config_.hidden_size),
        }),
        model_.token_embedding);
    if (!status) {
        return status;
    }
    status = alloc_view(
        weights_,
        make_shape({static_cast<int64_t>(config_.hidden_size)}),
        model_.final_norm);
    if (!status) {
        return status;
    }
    status = alloc_view(
        weights_,
        make_shape({
            static_cast<int64_t>(config_.vocab_size),
            static_cast<int64_t>(config_.hidden_size),
        }),
        model_.lm_head);
    if (!status) {
        return status;
    }

    const int64_t hidden = config_.hidden_size;
    const int64_t inter = config_.intermediate_size;
    const int64_t kv_dim = config_.n_kv_heads * config_.head_dim();

    for (LayerWeights& layer : model_.layers) {
        status = alloc_view(weights_, make_shape({hidden}), layer.attn_norm);
        if (!status) {
            return status;
        }
        status = alloc_view(weights_, make_shape({hidden, hidden}), layer.q_proj);
        if (!status) {
            return status;
        }
        status = alloc_view(weights_, make_shape({kv_dim, hidden}), layer.k_proj);
        if (!status) {
            return status;
        }
        status = alloc_view(weights_, make_shape({kv_dim, hidden}), layer.v_proj);
        if (!status) {
            return status;
        }
        status = alloc_view(weights_, make_shape({hidden, hidden}), layer.o_proj);
        if (!status) {
            return status;
        }
        status = alloc_view(weights_, make_shape({hidden}), layer.ffn_norm);
        if (!status) {
            return status;
        }
        status = alloc_view(weights_, make_shape({inter, hidden}), layer.gate_proj);
        if (!status) {
            return status;
        }
        status = alloc_view(weights_, make_shape({inter, hidden}), layer.up_proj);
        if (!status) {
            return status;
        }
        status = alloc_view(weights_, make_shape({hidden, inter}), layer.down_proj);
        if (!status) {
            return status;
        }
    }

    return Status::success();
}

Status LlamaInferEngine::init_kv_cache() {
    kv_cache_arena_.reset();
    const Shape shape = make_shape({
        static_cast<int64_t>(config_.n_layers),
        static_cast<int64_t>(config_.n_kv_heads),
        static_cast<int64_t>(max_seq_len_),
        static_cast<int64_t>(config_.head_dim()),
    });

    Status status = alloc_view(kv_cache_arena_, shape, cache_.keys);
    if (!status) {
        return status;
    }
    status = alloc_view(kv_cache_arena_, shape, cache_.values);
    if (!status) {
        return status;
    }

    cache_.seq_len = 0;
    cache_.max_seq_len = max_seq_len_;
    return Status::success();
}

void LlamaInferEngine::reset() {
    workspace_.reset();
    logits_ = {};
    kv_cache_arena_.reset();
    const Shape shape = make_shape({
        static_cast<int64_t>(config_.n_layers),
        static_cast<int64_t>(config_.n_kv_heads),
        static_cast<int64_t>(max_seq_len_),
        static_cast<int64_t>(config_.head_dim()),
    });
    cache_.keys = kv_cache_arena_.alloc(shape, DType::f32).value;
    cache_.values = kv_cache_arena_.alloc(shape, DType::f32).value;
    cache_.seq_len = 0;
    cache_.max_seq_len = max_seq_len_;
}

TensorView LlamaInferEngine::workspace_tensor(const Shape& shape) {
    return workspace_.alloc(shape, DType::f32).value;
}

void LlamaInferEngine::rebind_view_after_move(
    TensorView& view,
    const LlamaInferEngine& source) {
    if (view.arena == &source.weights_) {
        view.arena = &weights_;
    } else if (view.arena == &source.kv_cache_arena_) {
        view.arena = &kv_cache_arena_;
    } else if (view.arena == &source.workspace_) {
        view.arena = &workspace_;
    }
}

void LlamaInferEngine::rebind_views_after_move(const LlamaInferEngine& source) {
    rebind_view_after_move(model_.token_embedding, source);
    rebind_view_after_move(model_.final_norm, source);
    rebind_view_after_move(model_.lm_head, source);

    for (LayerWeights& layer : model_.layers) {
        rebind_view_after_move(layer.attn_norm, source);
        rebind_view_after_move(layer.q_proj, source);
        rebind_view_after_move(layer.k_proj, source);
        rebind_view_after_move(layer.v_proj, source);
        rebind_view_after_move(layer.o_proj, source);
        rebind_view_after_move(layer.ffn_norm, source);
        rebind_view_after_move(layer.gate_proj, source);
        rebind_view_after_move(layer.up_proj, source);
        rebind_view_after_move(layer.down_proj, source);
    }

    rebind_view_after_move(cache_.keys, source);
    rebind_view_after_move(cache_.values, source);
    rebind_view_after_move(logits_, source);
}

Status LlamaInferEngine::prefill(std::span<const TokenId> prompt, TokenId& next_token) {
    Status status = validate_prompt(prompt, config_, cache_.max_seq_len, 0);
    if (!status) {
        return status;
    }
    if (cache_.seq_len != 0) {
        return Status::invalid_argument_status("prefill requires an empty KV cache");
    }

    workspace_.reset();
    run_layers(prompt, 0, logits_);

    backend_->argmax(next_token, logits_);

    cache_.seq_len = static_cast<uint32_t>(prompt.size());
    return Status::success();
}

Status LlamaInferEngine::decode_one(TokenId token, TokenId& next_token) {
    const std::array<TokenId, 1> one_token = {token};
    Status status = validate_prompt(one_token, config_, cache_.max_seq_len, cache_.seq_len);
    if (!status) {
        return status;
    }

    workspace_.reset();
    run_layers(one_token, cache_.seq_len, logits_);

    backend_->argmax(next_token, logits_);

    ++cache_.seq_len;
    return Status::success();
}

Status LlamaInferEngine::generate(
    std::span<const TokenId> prompt,
    std::span<TokenId> output,
    const GenerateConfig& config,
    uint32_t& output_count) {
    output_count = 0;
    if (config.max_new_tokens == 0) {
        return Status::invalid_argument_status("max_new_tokens must be non-zero");
    }
    if (output.size() < prompt.size() + config.max_new_tokens) {
        return Status::invalid_argument_status("output buffer is too small");
    }
    Status status = validate_prompt(prompt, config_, cache_.max_seq_len, 0);
    if (!status) {
        return status;
    }
    if (prompt.size() + config.max_new_tokens > cache_.max_seq_len) {
        return Status::invalid_argument_status("generation exceeds KV cache capacity");
    }
    if (prompt.size() + config.max_new_tokens > config_.max_seq_len) {
        return Status::invalid_argument_status("generation exceeds model max sequence length");
    }

    reset();

    for (size_t i = 0; i < prompt.size(); ++i) {
        output[i] = prompt[i];
    }
    output_count = static_cast<uint32_t>(prompt.size());

    TokenId token = 0;
    status = prefill(prompt, token);
    if (!status) {
        return status;
    }

    output[output_count++] = token;
    if (config.stop_on_eos && token == config.eos_token_id) {
        return Status::success();
    }

    for (uint32_t i = 1; i < config.max_new_tokens; ++i) {
        TokenId next = 0;
        status = decode_one(token, next);
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

const LlamaConfig& LlamaInferEngine::config() const {
    return config_;
}

uint32_t LlamaInferEngine::seq_len() const {
    return cache_.seq_len;
}

uint32_t LlamaInferEngine::max_seq_len() const {
    return max_seq_len_;
}

void LlamaInferEngine::run_layers(
    std::span<const TokenId> tokens,
    uint32_t start_pos,
    TensorView& logits) {
    const uint32_t seq_len = static_cast<uint32_t>(tokens.size());
    TensorView hidden = workspace_tensor(make_shape({
        static_cast<int64_t>(seq_len),
        static_cast<int64_t>(config_.hidden_size),
    }));
    TensorView normed = workspace_tensor(make_shape({
        static_cast<int64_t>(seq_len),
        static_cast<int64_t>(config_.hidden_size),
    }));
    TensorView q = workspace_tensor(make_shape({
        static_cast<int64_t>(seq_len),
        static_cast<int64_t>(config_.n_heads),
        static_cast<int64_t>(config_.head_dim()),
    }));
    TensorView k = workspace_tensor(make_shape({
        static_cast<int64_t>(seq_len),
        static_cast<int64_t>(config_.n_kv_heads),
        static_cast<int64_t>(config_.head_dim()),
    }));
    TensorView v = workspace_tensor(make_shape({
        static_cast<int64_t>(seq_len),
        static_cast<int64_t>(config_.n_kv_heads),
        static_cast<int64_t>(config_.head_dim()),
    }));
    TensorView attn_out = workspace_tensor(make_shape({
        static_cast<int64_t>(seq_len),
        static_cast<int64_t>(config_.hidden_size),
    }));
    TensorView gate = workspace_tensor(make_shape({
        static_cast<int64_t>(seq_len),
        static_cast<int64_t>(config_.intermediate_size),
    }));
    TensorView up = workspace_tensor(make_shape({
        static_cast<int64_t>(seq_len),
        static_cast<int64_t>(config_.intermediate_size),
    }));
    TensorView swiglu = workspace_tensor(make_shape({
        static_cast<int64_t>(seq_len),
        static_cast<int64_t>(config_.intermediate_size),
    }));
    TensorView ffn_out = workspace_tensor(make_shape({
        static_cast<int64_t>(seq_len),
        static_cast<int64_t>(config_.hidden_size),
    }));

    backend_->embedding_out(
        hidden,
        model_.token_embedding,
        tokens);

    for (uint32_t i = 0; i < config_.n_layers; ++i) {
        LayerWeights& layer = model_.layers[i];

        backend_->rms_norm_out(normed, hidden, layer.attn_norm, config_.rms_eps);
        backend_->matmul_out(q, normed, layer.q_proj);
        backend_->matmul_out(k, normed, layer.k_proj);
        backend_->matmul_out(v, normed, layer.v_proj);
        backend_->rope_inplace(q, k, start_pos, config_.rope_theta);
        TensorView layer_k_cache = layer_cache_view(cache_.keys, i);
        TensorView layer_v_cache = layer_cache_view(cache_.values, i);
        backend_->attention_out(
            attn_out,
            q,
            k,
            v,
            layer_k_cache,
            layer_v_cache,
            start_pos,
            start_pos + seq_len);
        backend_->matmul_out(ffn_out, attn_out, layer.o_proj);
        backend_->add_inplace(ffn_out, hidden);
        backend_->rms_norm_out(normed, ffn_out, layer.ffn_norm, config_.rms_eps);
        backend_->matmul_out(gate, normed, layer.gate_proj);
        backend_->matmul_out(up, normed, layer.up_proj);
        backend_->swiglu_out(swiglu, gate, up);
        backend_->matmul_out(hidden, swiglu, layer.down_proj);
        backend_->add_inplace(hidden, ffn_out);
    }

    logits = workspace_tensor(make_shape({
        1,
        static_cast<int64_t>(config_.vocab_size),
    }));

    backend_->rms_norm_out(hidden, hidden, model_.final_norm, config_.rms_eps);
    TensorView last_hidden = last_token_view(hidden, seq_len - 1);
    backend_->matmul_out(logits, last_hidden, model_.lm_head);
}

}  // namespace tinyinfer
