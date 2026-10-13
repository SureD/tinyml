#include "tinyinfer/embedder.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace tinyinfer {
namespace {

constexpr size_t kTensorAlignmentSlack = 64;
constexpr float kNormalizeEps = 1e-12f;
// transformers EmbeddingGemma2Attention sets this explicitly. It is not 1/sqrt(head_dim).
constexpr float kAttentionScale = 1.0f;

bool checked_add(size_t a, size_t b, size_t& out) {
    if (a > std::numeric_limits<size_t>::max() - b) {
        return false;
    }
    out = a + b;
    return true;
}

bool checked_mul(size_t a, size_t b, size_t& out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

bool matryoshka_width(size_t width, uint32_t embedding_dim) {
    if (width == embedding_dim) {
        return true;
    }
    if (width > embedding_dim) {
        return false;
    }
    return width == 128 || width == 256 || width == 512;
}

Status alloc_dtype(MemoryArena& arena, const Shape& shape, DType dtype, TensorView& out) {
    Result<TensorView> view = arena.alloc(shape, dtype);
    if (!view.status) {
        return view.status;
    }
    out = view.value;
    return Status::success();
}

Status alloc_f32(MemoryArena& arena, const Shape& shape, TensorView& out) {
    return alloc_dtype(arena, shape, DType::f32, out);
}

Status require_f32(Backend& backend, const TensorView& view, float*& out) {
    if (view.dtype != DType::f32) {
        return Status::backend_error_status("embedder tensor is not float32");
    }
    void* bytes = backend.host_bytes(view);
    if (bytes == nullptr) {
        return Status::backend_error_status("embedder tensor is not a CPU f32 buffer");
    }
    out = static_cast<float*>(bytes);
    return Status::success();
}

Status require_i8(Backend& backend, const TensorView& view, int8_t*& out) {
    if (view.dtype != DType::i8) {
        return Status::backend_error_status("embedder tensor is not int8");
    }
    void* bytes = backend.host_bytes(view);
    if (bytes == nullptr) {
        return Status::backend_error_status("embedder tensor is not a CPU int8 buffer");
    }
    out = static_cast<int8_t*>(bytes);
    return Status::success();
}

int unpack_q(const int8_t* packed, int64_t index, int bits) {
    if (bits == 8) {
        return packed[index];
    }
    const uint8_t byte = static_cast<uint8_t>(packed[index / 2]);
    int nibble = (index & 1) != 0 ? (byte >> 4) : (byte & 0x0f);
    if ((nibble & 0x8) != 0) {
        nibble -= 16;
    }
    return nibble;
}

void store_q(int8_t* packed, int64_t index, int bits, int q) {
    if (bits == 8) {
        packed[index] = static_cast<int8_t>(q);
        return;
    }
    uint8_t* bytes = reinterpret_cast<uint8_t*>(packed);
    const uint8_t nibble = static_cast<uint8_t>(q) & 0x0f;
    if ((index & 1) == 0) {
        bytes[index / 2] = nibble;
    } else {
        bytes[index / 2] = static_cast<uint8_t>(bytes[index / 2] | (nibble << 4));
    }
}

int64_t packed_cols(int64_t cols, int bits) {
    if (bits == 8) {
        return cols;
    }
    return (cols + 1) / 2;
}

void quantize_matrix(
    const float* weight,
    int64_t rows,
    int64_t cols,
    int bits,
    std::vector<int8_t>& packed,
    std::vector<float>& scale,
    std::vector<int8_t>& zero) {
    const int qmin = bits == 8 ? -128 : -8;
    const int qmax = bits == 8 ? 127 : 7;
    const int64_t stride = packed_cols(cols, bits);
    packed.assign(static_cast<size_t>(rows * stride), 0);
    scale.assign(static_cast<size_t>(rows), 0.0f);
    zero.assign(static_cast<size_t>(rows), 0);
    for (int64_t row = 0; row < rows; ++row) {
        const float* src = weight + row * cols;
        int8_t* dst = packed.data() + row * stride;
        float min_v = src[0];
        float max_v = src[0];
        for (int64_t col = 1; col < cols; ++col) {
            min_v = std::min(min_v, src[col]);
            max_v = std::max(max_v, src[col]);
        }
        int zero_point = 0;
        float row_scale = 0.0f;
        if (max_v == min_v) {
            // (q - zero) * scale with q = 1 and zero = 0 reproduces the constant.
            row_scale = max_v;
            zero_point = 0;
            for (int64_t col = 0; col < cols; ++col) {
                store_q(dst, col, bits, 1);
            }
        } else {
            row_scale = (max_v - min_v) / static_cast<float>(qmax - qmin);
            float zp_f = std::round(static_cast<float>(qmin) - min_v / row_scale);
            if (zp_f < static_cast<float>(qmin)) {
                zp_f = static_cast<float>(qmin);
            }
            if (zp_f > static_cast<float>(qmax)) {
                zp_f = static_cast<float>(qmax);
            }
            zero_point = static_cast<int>(zp_f);
            for (int64_t col = 0; col < cols; ++col) {
                double scaled = static_cast<double>(src[col]) / static_cast<double>(row_scale);
                if (scaled > static_cast<double>(qmax - qmin)) {
                    scaled = static_cast<double>(qmax - qmin);
                }
                if (scaled < static_cast<double>(qmin - qmax)) {
                    scaled = static_cast<double>(qmin - qmax);
                }
                int q = static_cast<int>(std::lround(scaled)) + zero_point;
                if (q < qmin) {
                    q = qmin;
                }
                if (q > qmax) {
                    q = qmax;
                }
                store_q(dst, col, bits, q);
            }
        }
        scale[static_cast<size_t>(row)] = row_scale;
        zero[static_cast<size_t>(row)] = static_cast<int8_t>(zero_point);
    }
}

void gemm(
    float* out,
    const float* x,
    const float* weight,
    int64_t rows,
    int64_t k,
    int64_t cols) {
    // weight is [cols, k], row-major, matching a PyTorch linear weight.
    for (int64_t row = 0; row < rows; ++row) {
        const float* x_row = x + row * k;
        float* out_row = out + row * cols;
        for (int64_t col = 0; col < cols; ++col) {
            const float* w_row = weight + col * k;
            float sum = 0.0f;
            for (int64_t inner = 0; inner < k; ++inner) {
                sum += x_row[inner] * w_row[inner];
            }
            out_row[col] = sum;
        }
    }
}

void gemm_quant(
    float* out,
    const float* x,
    const int8_t* packed,
    const float* scale,
    const int8_t* zero,
    int64_t rows,
    int64_t k,
    int64_t cols,
    int bits) {
    // Dequantize a short run of the weight row, then accumulate float32.
    // The unpacked matrix is not stored.
    constexpr int64_t kTile = 32;
    for (int64_t row = 0; row < rows; ++row) {
        const float* x_row = x + row * k;
        float* out_row = out + row * cols;
        for (int64_t col = 0; col < cols; ++col) {
            const float row_scale = scale[col];
            const int row_zero = zero[col];
            const int8_t* row_q = packed + col * packed_cols(k, bits);
            float sum = 0.0f;
            for (int64_t inner = 0; inner < k;) {
                const int64_t tile = std::min(kTile, k - inner);
                float decoded[kTile];
                for (int64_t t = 0; t < tile; ++t) {
                    const int q = unpack_q(row_q, inner + t, bits);
                    decoded[t] = static_cast<float>(q - row_zero) * row_scale;
                }
                for (int64_t t = 0; t < tile; ++t) {
                    sum += x_row[inner + t] * decoded[t];
                }
                inner += tile;
            }
            out_row[col] = sum;
        }
    }
}

void rms_norm(
    float* out,
    const float* x,
    const float* weight,
    int64_t rows,
    int64_t dim,
    float eps) {
    // In-place is safe: the scale is a scalar computed before the row is written.
    for (int64_t row = 0; row < rows; ++row) {
        const float* src = x + row * dim;
        float sum_sq = 0.0f;
        for (int64_t i = 0; i < dim; ++i) {
            sum_sq += src[i] * src[i];
        }
        const float mean_sq = sum_sq / static_cast<float>(dim);
        const float scale = 1.0f / std::sqrt(mean_sq + eps);
        float* dst = out + row * dim;
        for (int64_t i = 0; i < dim; ++i) {
            float value = src[i] * scale;
            if (weight != nullptr) {
                value *= weight[i];
            }
            dst[i] = value;
        }
    }
}

void gelu_tanh(float* x, int64_t count) {
    // torch.nn.GELU(approximate="tanh"): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    const double k = std::sqrt(2.0 / 3.14159265358979323846);
    for (int64_t i = 0; i < count; ++i) {
        const double value = static_cast<double>(x[i]);
        const double inner = k * (value + 0.044715 * value * value * value);
        x[i] = static_cast<float>(0.5 * value * (1.0 + std::tanh(inner)));
    }
}

void apply_rope(
    float* states,
    int64_t seq,
    int64_t n_heads,
    int64_t head_dim,
    const float* inv_freq) {
    const int64_t half = head_dim / 2;
    for (int64_t t = 0; t < seq; ++t) {
        for (int64_t h = 0; h < n_heads; ++h) {
            float* row = states + (t * n_heads + h) * head_dim;
            for (int64_t i = 0; i < half; ++i) {
                const double freq = static_cast<double>(t) * static_cast<double>(inv_freq[i]);
                const float c = static_cast<float>(std::cos(freq));
                const float s = static_cast<float>(std::sin(freq));
                const float x1 = row[i];
                const float x2 = row[i + half];
                row[i] = x1 * c - x2 * s;
                row[i + half] = x2 * c + x1 * s;
            }
        }
    }
}

float dot_product(const float* a, const float* b, int64_t n) {
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

bool sliding_keeps(int64_t query, int64_t key, bool sliding, int64_t window) {
    if (!sliding) {
        return true;
    }
    const int64_t distance = query > key ? query - key : key - query;
    return distance <= window;
}

void attention(
    float* out,
    const float* q,
    const float* k,
    const float* v,
    float* weight,
    int64_t seq,
    int64_t n_heads,
    int64_t n_kv,
    int64_t head_dim,
    bool sliding,
    int64_t window) {
    // Softmax weights depend on Q and K, not on V. One vector per query head,
    // then a mix over V, is O(seq^2 * head_dim). Recomputing Q·K inside the
    // head-dim loop was O(seq^2 * head_dim^2). `weight` holds max_seq_len
    // floats from the workspace arena and is reused for every layer.
    const int64_t group = n_heads / n_kv;
    for (int64_t t = 0; t < seq; ++t) {
        for (int64_t h = 0; h < n_heads; ++h) {
            const int64_t kv_head = h / group;
            const float* q_row = q + (t * n_heads + h) * head_dim;
            float* out_row = out + (t * n_heads + h) * head_dim;

            float max_score = -std::numeric_limits<float>::infinity();
            for (int64_t p = 0; p < seq; ++p) {
                if (!sliding_keeps(t, p, sliding, window)) {
                    continue;
                }
                const float* k_row = k + (p * n_kv + kv_head) * head_dim;
                const float score = dot_product(q_row, k_row, head_dim) * kAttentionScale;
                weight[static_cast<size_t>(p)] = score;
                if (score > max_score) {
                    max_score = score;
                }
            }

            float denom = 0.0f;
            for (int64_t p = 0; p < seq; ++p) {
                if (!sliding_keeps(t, p, sliding, window)) {
                    weight[static_cast<size_t>(p)] = 0.0f;
                    continue;
                }
                const float coeff = static_cast<float>(
                    std::exp(static_cast<double>(weight[static_cast<size_t>(p)] - max_score)));
                weight[static_cast<size_t>(p)] = coeff;
                denom += coeff;
            }
            const float inv_denom = 1.0f / denom;
            for (int64_t p = 0; p < seq; ++p) {
                if (sliding_keeps(t, p, sliding, window)) {
                    weight[static_cast<size_t>(p)] *= inv_denom;
                }
            }

            for (int64_t d = 0; d < head_dim; ++d) {
                float sum = 0.0f;
                for (int64_t p = 0; p < seq; ++p) {
                    if (!sliding_keeps(t, p, sliding, window)) {
                        continue;
                    }
                    const float* v_row = v + (p * n_kv + kv_head) * head_dim;
                    sum += weight[static_cast<size_t>(p)] * v_row[d];
                }
                out_row[d] = sum;
            }
        }
    }
}

void mean_pool(float* out, const float* x, int64_t seq, int64_t dim) {
    for (int64_t d = 0; d < dim; ++d) {
        float sum = 0.0f;
        for (int64_t t = 0; t < seq; ++t) {
            sum += x[t * dim + d];
        }
        out[d] = sum / static_cast<float>(seq);
    }
}

void l2_normalize(float* x, int64_t dim) {
    float sum_sq = 0.0f;
    for (int64_t i = 0; i < dim; ++i) {
        sum_sq += x[i] * x[i];
    }
    float norm = std::sqrt(sum_sq);
    if (norm < kNormalizeEps) {
        norm = kNormalizeEps;
    }
    const float inv = 1.0f / norm;
    for (int64_t i = 0; i < dim; ++i) {
        x[i] *= inv;
    }
}

bool all_finite(const float* x, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        if (!std::isfinite(x[i])) {
            return false;
        }
    }
    return true;
}

std::vector<float> make_inv_freq(float theta, uint32_t head_dim) {
    const uint32_t half = head_dim / 2;
    std::vector<float> inv(half);
    for (uint32_t i = 0; i < half; ++i) {
        const double exponent =
            (2.0 * static_cast<double>(i)) / static_cast<double>(head_dim);
        const double value = 1.0 / std::pow(static_cast<double>(theta), exponent);
        inv[i] = static_cast<float>(value);
    }
    return inv;
}

size_t weight_tensor_count(const EmbeddingGemma2Config& config) {
    constexpr size_t kGlobalTensors = 5;
    constexpr size_t kLayerTensors = 17;
    return kGlobalTensors + static_cast<size_t>(config.n_layers) * kLayerTensors;
}

size_t weight_elems(const EmbeddingGemma2Config& config) {
    const size_t hidden = config.hidden_size;
    const size_t inter = config.intermediate_size;
    const size_t ple = config.ple_dim;
    size_t total =
        static_cast<size_t>(config.vocab_size) * hidden +
        hidden +
        static_cast<size_t>(config.embedding_dim) * hidden +
        static_cast<size_t>(config.n_layers) * ple * hidden +
        ple;

    for (uint32_t layer = 0; layer < config.n_layers; ++layer) {
        const size_t q_dim = config.q_dim(layer);
        const size_t kv_dim = config.kv_dim(layer);
        const size_t head_dim = config.layers[layer].head_dim;
        total +=
            4 * hidden +          // four residual norms
            q_dim * hidden +      // q
            kv_dim * hidden +     // k
            kv_dim * hidden +     // v
            hidden * q_dim +      // o
            head_dim +            // q_norm
            head_dim +            // k_norm
            inter * hidden +      // gate
            inter * hidden +      // up
            hidden * inter +      // down
            ple * hidden +        // ple gate
            hidden * ple +        // ple proj
            hidden +              // ple norm
            1;                    // layer scalar
    }
    return total;
}

constexpr size_t kWorkspaceTensors = 14;

size_t workspace_elems(const EmbeddingGemma2Config& config, uint32_t max_seq_len) {
    const size_t seq = max_seq_len;
    const size_t hidden = config.hidden_size;
    const size_t inter = config.intermediate_size;
    const size_t ple = config.ple_dim;
    const size_t embed = config.embedding_dim;
    const size_t max_q = config.max_q_dim();
    const size_t max_kv = config.max_kv_dim();
    return 3 * seq * hidden +
        2 * seq * max_q +
        2 * seq * max_kv +
        2 * seq * inter +
        seq * ple +
        seq * static_cast<size_t>(config.n_layers) * ple +
        seq * embed +
        embed +
        seq;
}

}  // namespace

uint32_t EmbeddingGemma2Config::q_dim(uint32_t layer) const {
    return n_heads * layers[layer].head_dim;
}

uint32_t EmbeddingGemma2Config::kv_dim(uint32_t layer) const {
    return layers[layer].n_kv_heads * layers[layer].head_dim;
}

uint32_t EmbeddingGemma2Config::max_q_dim() const {
    uint32_t max_dim = 0;
    for (uint32_t layer = 0; layer < n_layers && layer < layers.size(); ++layer) {
        const uint32_t dim = q_dim(layer);
        if (dim > max_dim) {
            max_dim = dim;
        }
    }
    return max_dim;
}

uint32_t EmbeddingGemma2Config::max_kv_dim() const {
    uint32_t max_dim = 0;
    for (uint32_t layer = 0; layer < n_layers && layer < layers.size(); ++layer) {
        const uint32_t dim = kv_dim(layer);
        if (dim > max_dim) {
            max_dim = dim;
        }
    }
    return max_dim;
}

Status EmbeddingGemma2Config::validate() const {
    if (n_layers == 0 || layers.size() != n_layers) {
        return Status::invalid_config_status("layer table does not match layer count");
    }
    if (hidden_size == 0 || intermediate_size == 0 || n_heads == 0 ||
        vocab_size == 0 || embedding_dim == 0 || ple_dim == 0 ||
        max_position_embeddings == 0) {
        return Status::invalid_config_status("embedding config has a zero extent");
    }
    if (!std::isfinite(rms_eps) || rms_eps <= 0.0f) {
        return Status::invalid_config_status("rms norm epsilon must be positive");
    }
    if (ple_dim != 0 && n_layers > std::numeric_limits<uint32_t>::max() / ple_dim) {
        return Status::invalid_config_status("per-layer embedding width overflows");
    }

    for (uint32_t layer = 0; layer < n_layers; ++layer) {
        const EmbeddingGemma2Layer& spec = layers[layer];
        if (spec.head_dim == 0 || (spec.head_dim % 2) != 0 || spec.n_kv_heads == 0) {
            return Status::invalid_config_status("invalid attention head shape");
        }
        if (n_heads % spec.n_kv_heads != 0) {
            return Status::invalid_config_status("query heads are not divisible by kv heads");
        }
        if (spec.head_dim != 0 && n_heads > std::numeric_limits<uint32_t>::max() / spec.head_dim) {
            return Status::invalid_config_status("query width overflows");
        }
        if (!std::isfinite(spec.rope_theta) || spec.rope_theta <= 0.0f) {
            return Status::invalid_config_status("rope theta must be positive");
        }
    }
    return Status::success();
}

EmbeddingGemma2Engine::EmbeddingGemma2Engine(
    Backend& backend,
    const EmbeddingGemma2Config& config,
    uint32_t max_seq_len)
    : backend_(&backend),
      config_(config),
      max_seq_len_(max_seq_len) {}

EmbeddingGemma2Engine::EmbeddingGemma2Engine(EmbeddingGemma2Engine&& other) noexcept
    : backend_(other.backend_),
      config_(std::move(other.config_)),
      max_seq_len_(other.max_seq_len_),
      weights_loaded_(other.weights_loaded_),
      quant_(other.quant_),
      weights_(std::move(other.weights_)),
      workspace_(std::move(other.workspace_)),
      model_(std::move(other.model_)),
      work_(other.work_),
      rope_inv_freq_(std::move(other.rope_inv_freq_)) {
    rebind_views_after_move(other);
    other.backend_ = nullptr;
    other.max_seq_len_ = 0;
    other.weights_loaded_ = false;
    other.quant_ = EmbeddingGemma2Quant::f32;
    other.work_ = {};
}

EmbeddingGemma2Engine& EmbeddingGemma2Engine::operator=(EmbeddingGemma2Engine&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    backend_ = other.backend_;
    config_ = std::move(other.config_);
    max_seq_len_ = other.max_seq_len_;
    weights_loaded_ = other.weights_loaded_;
    quant_ = other.quant_;
    weights_ = std::move(other.weights_);
    workspace_ = std::move(other.workspace_);
    model_ = std::move(other.model_);
    work_ = other.work_;
    rope_inv_freq_ = std::move(other.rope_inv_freq_);
    rebind_views_after_move(other);
    other.backend_ = nullptr;
    other.max_seq_len_ = 0;
    other.weights_loaded_ = false;
    other.quant_ = EmbeddingGemma2Quant::f32;
    other.work_ = {};
    return *this;
}

Result<EmbeddingGemma2Engine> EmbeddingGemma2Engine::create(
    Backend& backend,
    const EmbeddingGemma2Config& config,
    uint32_t max_seq_len) {
    EmbeddingGemma2Engine engine(backend, config, max_seq_len);
    Status status = engine.init();
    if (!status) {
        return {status, EmbeddingGemma2Engine()};
    }
    return {Status::success(), std::move(engine)};
}

Status EmbeddingGemma2Engine::init() {
    Status status = config_.validate();
    if (!status) {
        return status;
    }
    if (max_seq_len_ == 0 || max_seq_len_ > config_.max_position_embeddings) {
        return Status::invalid_argument_status("invalid embedder sequence capacity");
    }

    size_t weight_bytes = 0;
    if (!checked_mul(weight_elems(config_), dtype_size(DType::f32), weight_bytes) ||
        !checked_add(weight_bytes, weight_tensor_count(config_) * kTensorAlignmentSlack, weight_bytes)) {
        return Status::invalid_argument_status("weight arena size overflows");
    }
    size_t workspace_bytes = 0;
    if (!checked_mul(workspace_elems(config_, max_seq_len_), dtype_size(DType::f32), workspace_bytes) ||
        !checked_add(
            workspace_bytes,
            kWorkspaceTensors * kTensorAlignmentSlack,
            workspace_bytes)) {
        return Status::invalid_argument_status("workspace arena size overflows");
    }

    status = backend_->alloc_arena(weights_, weight_bytes, MemoryKind::weights);
    if (!status) {
        return status;
    }
    status = backend_->alloc_arena(workspace_, workspace_bytes, MemoryKind::workspace);
    if (!status) {
        return status;
    }
    status = bind_weights();
    if (!status) {
        return status;
    }
    status = bind_workspace();
    if (!status) {
        return status;
    }

    rope_inv_freq_.resize(config_.n_layers);
    for (uint32_t layer = 0; layer < config_.n_layers; ++layer) {
        rope_inv_freq_[layer] = make_inv_freq(
            config_.layers[layer].rope_theta,
            config_.layers[layer].head_dim);
    }
    return Status::success();
}

Status EmbeddingGemma2Engine::bind_weights() {
    const int64_t hidden = config_.hidden_size;
    const int64_t inter = config_.intermediate_size;
    const int64_t ple = config_.ple_dim;
    const int64_t embed = config_.embedding_dim;
    const int64_t layers = config_.n_layers;

    auto alloc_matrix = [&](int64_t rows, int64_t cols, Matrix& out) -> Status {
        out.rows = rows;
        out.cols = cols;
        out.scale = {};
        out.zero = {};
        return alloc_f32(weights_, make_shape({rows, cols}), out.weight);
    };

    Status status = alloc_matrix(config_.vocab_size, hidden, model_.embed_tokens);
    if (!status) {
        return status;
    }
    status = alloc_f32(weights_, make_shape({hidden}), model_.final_norm);
    if (!status) {
        return status;
    }
    status = alloc_matrix(embed, hidden, model_.embedding_projection);
    if (!status) {
        return status;
    }
    status = alloc_matrix(layers * ple, hidden, model_.ple_projection);
    if (!status) {
        return status;
    }
    status = alloc_f32(weights_, make_shape({ple}), model_.ple_norm);
    if (!status) {
        return status;
    }

    model_.layers.resize(config_.n_layers);
    for (uint32_t i = 0; i < config_.n_layers; ++i) {
        LayerWeights& layer = model_.layers[i];
        const int64_t q_dim = config_.q_dim(i);
        const int64_t kv_dim = config_.kv_dim(i);
        const int64_t head_dim = config_.layers[i].head_dim;

        status = alloc_f32(weights_, make_shape({hidden}), layer.input_layernorm);
        if (!status) {
            return status;
        }
        status = alloc_f32(weights_, make_shape({hidden}), layer.post_attention_layernorm);
        if (!status) {
            return status;
        }
        status = alloc_f32(weights_, make_shape({hidden}), layer.pre_feedforward_layernorm);
        if (!status) {
            return status;
        }
        status = alloc_f32(weights_, make_shape({hidden}), layer.post_feedforward_layernorm);
        if (!status) {
            return status;
        }
        status = alloc_matrix(q_dim, hidden, layer.q_proj);
        if (!status) {
            return status;
        }
        status = alloc_matrix(kv_dim, hidden, layer.k_proj);
        if (!status) {
            return status;
        }
        status = alloc_matrix(kv_dim, hidden, layer.v_proj);
        if (!status) {
            return status;
        }
        status = alloc_matrix(hidden, q_dim, layer.o_proj);
        if (!status) {
            return status;
        }
        status = alloc_f32(weights_, make_shape({head_dim}), layer.q_norm);
        if (!status) {
            return status;
        }
        status = alloc_f32(weights_, make_shape({head_dim}), layer.k_norm);
        if (!status) {
            return status;
        }
        status = alloc_matrix(inter, hidden, layer.gate_proj);
        if (!status) {
            return status;
        }
        status = alloc_matrix(inter, hidden, layer.up_proj);
        if (!status) {
            return status;
        }
        status = alloc_matrix(hidden, inter, layer.down_proj);
        if (!status) {
            return status;
        }
        status = alloc_matrix(ple, hidden, layer.ple_gate);
        if (!status) {
            return status;
        }
        status = alloc_matrix(hidden, ple, layer.ple_proj);
        if (!status) {
            return status;
        }
        status = alloc_f32(weights_, make_shape({hidden}), layer.ple_norm);
        if (!status) {
            return status;
        }
        status = alloc_f32(weights_, make_shape({1}), layer.layer_scalar);
        if (!status) {
            return status;
        }
    }
    return Status::success();
}

Status EmbeddingGemma2Engine::bind_workspace() {
    const int64_t seq = max_seq_len_;
    const int64_t hidden = config_.hidden_size;
    const int64_t inter = config_.intermediate_size;
    const int64_t ple = config_.ple_dim;
    const int64_t embed = config_.embedding_dim;
    const int64_t max_q = config_.max_q_dim();
    const int64_t max_kv = config_.max_kv_dim();
    const int64_t layers = config_.n_layers;

    Status status = alloc_f32(workspace_, make_shape({seq, hidden}), work_.hidden);
    if (!status) {
        return status;
    }
    status = alloc_f32(workspace_, make_shape({seq, hidden}), work_.residual);
    if (!status) {
        return status;
    }
    status = alloc_f32(workspace_, make_shape({seq, hidden}), work_.normed);
    if (!status) {
        return status;
    }
    status = alloc_f32(workspace_, make_shape({seq, max_q}), work_.q);
    if (!status) {
        return status;
    }
    status = alloc_f32(workspace_, make_shape({seq, max_kv}), work_.k);
    if (!status) {
        return status;
    }
    status = alloc_f32(workspace_, make_shape({seq, max_kv}), work_.v);
    if (!status) {
        return status;
    }
    status = alloc_f32(workspace_, make_shape({seq, max_q}), work_.attn);
    if (!status) {
        return status;
    }
    status = alloc_f32(workspace_, make_shape({seq}), work_.attn_score);
    if (!status) {
        return status;
    }
    status = alloc_f32(workspace_, make_shape({seq, inter}), work_.gate);
    if (!status) {
        return status;
    }
    status = alloc_f32(workspace_, make_shape({seq, inter}), work_.up);
    if (!status) {
        return status;
    }
    status = alloc_f32(workspace_, make_shape({seq, ple}), work_.ple_h);
    if (!status) {
        return status;
    }
    status = alloc_f32(workspace_, make_shape({seq, layers, ple}), work_.ple);
    if (!status) {
        return status;
    }
    status = alloc_f32(workspace_, make_shape({seq, embed}), work_.projected);
    if (!status) {
        return status;
    }
    status = alloc_f32(workspace_, make_shape({embed}), work_.pooled);
    if (!status) {
        return status;
    }
    return Status::success();
}

void EmbeddingGemma2Engine::rebind_view(TensorView& view, const EmbeddingGemma2Engine& source) {
    if (view.arena == &source.weights_) {
        view.arena = &weights_;
    } else if (view.arena == &source.workspace_) {
        view.arena = &workspace_;
    }
}

void EmbeddingGemma2Engine::rebind_matrix(Matrix& matrix, const EmbeddingGemma2Engine& source) {
    rebind_view(matrix.weight, source);
    rebind_view(matrix.scale, source);
    rebind_view(matrix.zero, source);
}

void EmbeddingGemma2Engine::rebind_views_after_move(const EmbeddingGemma2Engine& source) {
    rebind_matrix(model_.embed_tokens, source);
    rebind_view(model_.final_norm, source);
    rebind_matrix(model_.embedding_projection, source);
    rebind_matrix(model_.ple_projection, source);
    rebind_view(model_.ple_norm, source);
    for (LayerWeights& layer : model_.layers) {
        rebind_view(layer.input_layernorm, source);
        rebind_view(layer.post_attention_layernorm, source);
        rebind_view(layer.pre_feedforward_layernorm, source);
        rebind_view(layer.post_feedforward_layernorm, source);
        rebind_matrix(layer.q_proj, source);
        rebind_matrix(layer.k_proj, source);
        rebind_matrix(layer.v_proj, source);
        rebind_matrix(layer.o_proj, source);
        rebind_view(layer.q_norm, source);
        rebind_view(layer.k_norm, source);
        rebind_matrix(layer.gate_proj, source);
        rebind_matrix(layer.up_proj, source);
        rebind_matrix(layer.down_proj, source);
        rebind_matrix(layer.ple_gate, source);
        rebind_matrix(layer.ple_proj, source);
        rebind_view(layer.ple_norm, source);
        rebind_view(layer.layer_scalar, source);
    }
    rebind_view(work_.hidden, source);
    rebind_view(work_.residual, source);
    rebind_view(work_.normed, source);
    rebind_view(work_.q, source);
    rebind_view(work_.k, source);
    rebind_view(work_.v, source);
    rebind_view(work_.attn, source);
    rebind_view(work_.attn_score, source);
    rebind_view(work_.gate, source);
    rebind_view(work_.up, source);
    rebind_view(work_.ple_h, source);
    rebind_view(work_.ple, source);
    rebind_view(work_.projected, source);
    rebind_view(work_.pooled, source);
}

const EmbeddingGemma2Config& EmbeddingGemma2Engine::config() const {
    return config_;
}

uint32_t EmbeddingGemma2Engine::max_seq_len() const {
    return max_seq_len_;
}

size_t EmbeddingGemma2Engine::weight_arena_bytes() const {
    return weights_.capacity();
}

size_t EmbeddingGemma2Engine::workspace_arena_bytes() const {
    return workspace_.capacity();
}

bool EmbeddingGemma2Engine::weights_loaded() const {
    return weights_loaded_;
}

EmbeddingGemma2Quant EmbeddingGemma2Engine::weight_quant() const {
    return quant_;
}

Status EmbeddingGemma2Engine::quantize_weights(EmbeddingGemma2Quant quant) {
    if (backend_ == nullptr) {
        return Status::invalid_argument_status("embedder is empty");
    }
    if (!weights_loaded_ || quant_ != EmbeddingGemma2Quant::f32) {
        return Status::invalid_argument_status("quantize requires loaded float32 weights");
    }
    int bits = 0;
    if (quant == EmbeddingGemma2Quant::int8) {
        bits = 8;
    } else if (quant == EmbeddingGemma2Quant::int4) {
        bits = 4;
    } else {
        return Status::invalid_argument_status("quantize target must be int8 or int4");
    }

    // Per matrix row, not a LiteRT file. Each 2-D weight is read as float32,
    // packed, then the float copy is dropped. 1-D norms stay float32.
    struct PackedMatrix {
        Matrix* matrix = nullptr;
        std::vector<int8_t> packed;
        std::vector<float> scale;
        std::vector<int8_t> zero;
    };
    struct SavedVector {
        TensorView* view = nullptr;
        Shape shape;
        std::vector<float> values;
    };

    std::vector<Matrix*> matrices;
    auto push_matrix = [&](Matrix& matrix) { matrices.push_back(&matrix); };
    push_matrix(model_.embed_tokens);
    push_matrix(model_.embedding_projection);
    push_matrix(model_.ple_projection);
    for (LayerWeights& layer : model_.layers) {
        push_matrix(layer.q_proj);
        push_matrix(layer.k_proj);
        push_matrix(layer.v_proj);
        push_matrix(layer.o_proj);
        push_matrix(layer.gate_proj);
        push_matrix(layer.up_proj);
        push_matrix(layer.down_proj);
        push_matrix(layer.ple_gate);
        push_matrix(layer.ple_proj);
    }

    std::vector<TensorView*> vectors;
    auto push_vector = [&](TensorView& view) { vectors.push_back(&view); };
    push_vector(model_.final_norm);
    push_vector(model_.ple_norm);
    for (LayerWeights& layer : model_.layers) {
        push_vector(layer.input_layernorm);
        push_vector(layer.post_attention_layernorm);
        push_vector(layer.pre_feedforward_layernorm);
        push_vector(layer.post_feedforward_layernorm);
        push_vector(layer.q_norm);
        push_vector(layer.k_norm);
        push_vector(layer.ple_norm);
        push_vector(layer.layer_scalar);
    }

    std::vector<PackedMatrix> packed(matrices.size());
    for (size_t i = 0; i < matrices.size(); ++i) {
        Matrix& matrix = *matrices[i];
        if (matrix.rows <= 0 || matrix.cols <= 0 || matrix.weight.dtype != DType::f32) {
            return Status::backend_error_status("weight matrix is not float32");
        }
        const int64_t count = matrix.rows * matrix.cols;
        if (count <= 0 || static_cast<size_t>(count) != static_cast<size_t>(matrix.weight.numel())) {
            return Status::backend_error_status("weight matrix shape does not match its view");
        }
        std::vector<float> values(static_cast<size_t>(count));
        Status status = backend_->copy_to_host(
            values.data(),
            matrix.weight,
            matrix.weight.logical_nbytes());
        if (!status) {
            return status;
        }
        for (float value : values) {
            if (!std::isfinite(value)) {
                return Status::invalid_argument_status("weight matrix is not finite");
            }
        }
        packed[i].matrix = &matrix;
        quantize_matrix(
            values.data(),
            matrix.rows,
            matrix.cols,
            bits,
            packed[i].packed,
            packed[i].scale,
            packed[i].zero);
    }

    std::vector<SavedVector> saved(vectors.size());
    for (size_t i = 0; i < vectors.size(); ++i) {
        TensorView& view = *vectors[i];
        if (view.dtype != DType::f32 || view.numel() <= 0) {
            return Status::backend_error_status("norm weight is not float32");
        }
        saved[i].view = &view;
        saved[i].shape = view.shape;
        saved[i].values.resize(static_cast<size_t>(view.numel()));
        Status status = backend_->copy_to_host(
            saved[i].values.data(),
            view,
            view.logical_nbytes());
        if (!status) {
            return status;
        }
    }

    size_t bytes = 0;
    auto add_tensor = [&](size_t nbytes) -> bool {
        return checked_add(bytes, nbytes, bytes) &&
            checked_add(bytes, kTensorAlignmentSlack, bytes);
    };
    for (const PackedMatrix& item : packed) {
        size_t scale_bytes = 0;
        if (!checked_mul(item.scale.size(), sizeof(float), scale_bytes) ||
            !add_tensor(item.packed.size()) ||
            !add_tensor(scale_bytes) ||
            !add_tensor(item.zero.size())) {
            return Status::invalid_argument_status("quantized weight arena size overflows");
        }
    }
    for (const SavedVector& item : saved) {
        size_t vector_bytes = 0;
        if (!checked_mul(item.values.size(), sizeof(float), vector_bytes) ||
            !add_tensor(vector_bytes)) {
            return Status::invalid_argument_status("quantized weight arena size overflows");
        }
    }

    Status status = backend_->alloc_arena(weights_, bytes, MemoryKind::weights);
    if (!status) {
        return status;
    }

    auto fail = [&](Status error) -> Status {
        weights_loaded_ = false;
        quant_ = EmbeddingGemma2Quant::f32;
        return error;
    };

    for (PackedMatrix& item : packed) {
        const int64_t stride = packed_cols(item.matrix->cols, bits);
        status = alloc_dtype(
            weights_,
            make_shape({item.matrix->rows, stride}),
            DType::i8,
            item.matrix->weight);
        if (!status) {
            return fail(status);
        }
        status = alloc_dtype(
            weights_,
            make_shape({item.matrix->rows}),
            DType::f32,
            item.matrix->scale);
        if (!status) {
            return fail(status);
        }
        status = alloc_dtype(
            weights_,
            make_shape({item.matrix->rows}),
            DType::i8,
            item.matrix->zero);
        if (!status) {
            return fail(status);
        }
        status = backend_->copy_from_host(
            item.matrix->weight,
            item.packed.data(),
            item.packed.size());
        if (!status) {
            return fail(status);
        }
        status = backend_->copy_from_host(
            item.matrix->scale,
            item.scale.data(),
            item.scale.size() * sizeof(float));
        if (!status) {
            return fail(status);
        }
        status = backend_->copy_from_host(
            item.matrix->zero,
            item.zero.data(),
            item.zero.size());
        if (!status) {
            return fail(status);
        }
    }

    for (SavedVector& item : saved) {
        status = alloc_dtype(weights_, item.shape, DType::f32, *item.view);
        if (!status) {
            return fail(status);
        }
        status = backend_->copy_from_host(
            *item.view,
            item.values.data(),
            item.values.size() * sizeof(float));
        if (!status) {
            return fail(status);
        }
    }

    quant_ = quant;
    weights_loaded_ = true;
    return Status::success();
}

Status EmbeddingGemma2Engine::embed(
    std::span<const uint32_t> tokens,
    std::span<float> embedding) {
    if (backend_ == nullptr) {
        return Status::invalid_argument_status("embedder is empty");
    }
    if (backend_->device().type != DeviceType::cpu) {
        return Status::unimplemented_status("EmbeddingGemma 2 embed is implemented for the CPU backend");
    }
    if (!weights_loaded_) {
        return Status::invalid_argument_status("embedder weights are not loaded");
    }
    if (tokens.empty()) {
        return Status::invalid_argument_status("token sequence is empty");
    }
    if (tokens.size() > max_seq_len_) {
        return Status::invalid_argument_status("token sequence exceeds embedder capacity");
    }
    if (!matryoshka_width(embedding.size(), config_.embedding_dim)) {
        return Status::invalid_argument_status("embedding width is not the model width or a matryoshka width");
    }
    for (uint32_t token : tokens) {
        if (token >= config_.vocab_size) {
            return Status::invalid_argument_status("token id exceeds vocab size");
        }
    }
    return forward(tokens, embedding);
}

Status EmbeddingGemma2Engine::forward(
    std::span<const uint32_t> tokens,
    std::span<float> embedding) {
    const int64_t seq = static_cast<int64_t>(tokens.size());
    const int64_t hidden = config_.hidden_size;
    const int64_t inter = config_.intermediate_size;
    const int64_t ple_dim = config_.ple_dim;
    const int64_t embed_dim = config_.embedding_dim;
    const int64_t n_layers = config_.n_layers;
    const int64_t n_heads = config_.n_heads;
    const float eps = config_.rms_eps;
    const float embed_scale = std::sqrt(static_cast<float>(hidden));
    const float ple_scale = 1.0f / embed_scale;

    float* hidden_states = nullptr;
    float* residual = nullptr;
    float* normed = nullptr;
    float* q = nullptr;
    float* k = nullptr;
    float* v = nullptr;
    float* attn = nullptr;
    float* attn_score = nullptr;
    float* gate = nullptr;
    float* up = nullptr;
    float* ple_h = nullptr;
    float* ple = nullptr;
    float* projected = nullptr;
    float* pooled = nullptr;
    float* ple_norm_weight = nullptr;
    float* final_norm = nullptr;

    Status status = require_f32(*backend_, work_.hidden, hidden_states);
    if (!status) {
        return status;
    }
    status = require_f32(*backend_, work_.residual, residual);
    if (!status) {
        return status;
    }
    status = require_f32(*backend_, work_.normed, normed);
    if (!status) {
        return status;
    }
    status = require_f32(*backend_, work_.q, q);
    if (!status) {
        return status;
    }
    status = require_f32(*backend_, work_.k, k);
    if (!status) {
        return status;
    }
    status = require_f32(*backend_, work_.v, v);
    if (!status) {
        return status;
    }
    status = require_f32(*backend_, work_.attn, attn);
    if (!status) {
        return status;
    }
    status = require_f32(*backend_, work_.attn_score, attn_score);
    if (!status) {
        return status;
    }
    status = require_f32(*backend_, work_.gate, gate);
    if (!status) {
        return status;
    }
    status = require_f32(*backend_, work_.up, up);
    if (!status) {
        return status;
    }
    status = require_f32(*backend_, work_.ple_h, ple_h);
    if (!status) {
        return status;
    }
    status = require_f32(*backend_, work_.ple, ple);
    if (!status) {
        return status;
    }
    status = require_f32(*backend_, work_.projected, projected);
    if (!status) {
        return status;
    }
    status = require_f32(*backend_, work_.pooled, pooled);
    if (!status) {
        return status;
    }
    status = require_f32(*backend_, model_.ple_norm, ple_norm_weight);
    if (!status) {
        return status;
    }
    status = require_f32(*backend_, model_.final_norm, final_norm);
    if (!status) {
        return status;
    }

    const int quant_bits = quant_ == EmbeddingGemma2Quant::int8 ? 8 : 4;
    auto linear = [&](float* dst, const float* src, const Matrix& matrix, int64_t m, int64_t k, int64_t n) -> Status {
        if (matrix.rows != n || matrix.cols != k) {
            return Status::backend_error_status("matrix shape does not match the linear");
        }
        if (quant_ == EmbeddingGemma2Quant::f32) {
            float* weight = nullptr;
            Status linear_status = require_f32(*backend_, matrix.weight, weight);
            if (!linear_status) {
                return linear_status;
            }
            gemm(dst, src, weight, m, k, n);
            return Status::success();
        }
        int8_t* packed = nullptr;
        float* scale = nullptr;
        int8_t* zero = nullptr;
        Status linear_status = require_i8(*backend_, matrix.weight, packed);
        if (!linear_status) {
            return linear_status;
        }
        linear_status = require_f32(*backend_, matrix.scale, scale);
        if (!linear_status) {
            return linear_status;
        }
        linear_status = require_i8(*backend_, matrix.zero, zero);
        if (!linear_status) {
            return linear_status;
        }
        gemm_quant(dst, src, packed, scale, zero, m, k, n, quant_bits);
        return Status::success();
    };

    if (quant_ == EmbeddingGemma2Quant::f32) {
        float* table = nullptr;
        status = require_f32(*backend_, model_.embed_tokens.weight, table);
        if (!status) {
            return status;
        }
        for (int64_t t = 0; t < seq; ++t) {
            const float* src = table + static_cast<int64_t>(tokens[static_cast<size_t>(t)]) * hidden;
            float* dst = hidden_states + t * hidden;
            for (int64_t i = 0; i < hidden; ++i) {
                dst[i] = src[i] * embed_scale;
            }
        }
    } else {
        int8_t* packed = nullptr;
        float* scale = nullptr;
        int8_t* zero = nullptr;
        status = require_i8(*backend_, model_.embed_tokens.weight, packed);
        if (!status) {
            return status;
        }
        status = require_f32(*backend_, model_.embed_tokens.scale, scale);
        if (!status) {
            return status;
        }
        status = require_i8(*backend_, model_.embed_tokens.zero, zero);
        if (!status) {
            return status;
        }
        const int64_t stride = packed_cols(hidden, quant_bits);
        for (int64_t t = 0; t < seq; ++t) {
            const int64_t id = static_cast<int64_t>(tokens[static_cast<size_t>(t)]);
            const int8_t* row = packed + id * stride;
            float* dst = hidden_states + t * hidden;
            for (int64_t i = 0; i < hidden; ++i) {
                const int q = unpack_q(row, i, quant_bits);
                dst[i] = static_cast<float>(q - zero[id]) * scale[id] * embed_scale;
            }
        }
    }

    status = linear(ple, hidden_states, model_.ple_projection, seq, hidden, n_layers * ple_dim);
    if (!status) {
        return status;
    }
    const int64_t ple_count = seq * n_layers * ple_dim;
    for (int64_t i = 0; i < ple_count; ++i) {
        ple[i] *= ple_scale;
    }
    rms_norm(ple, ple, ple_norm_weight, seq * n_layers, ple_dim, eps);

    for (int64_t layer_index = 0; layer_index < n_layers; ++layer_index) {
        const EmbeddingGemma2Layer& spec = config_.layers[static_cast<size_t>(layer_index)];
        const LayerWeights& layer = model_.layers[static_cast<size_t>(layer_index)];
        const int64_t head_dim = spec.head_dim;
        const int64_t n_kv = spec.n_kv_heads;
        const int64_t q_dim = n_heads * head_dim;
        const int64_t kv_dim = n_kv * head_dim;

        float* input_ln = nullptr;
        float* post_attn_ln = nullptr;
        float* pre_ffn_ln = nullptr;
        float* post_ffn_ln = nullptr;
        float* q_norm = nullptr;
        float* k_norm = nullptr;
        float* ple_post_norm = nullptr;
        float* layer_scalar = nullptr;

        status = require_f32(*backend_, layer.input_layernorm, input_ln);
        if (!status) {
            return status;
        }
        status = require_f32(*backend_, layer.post_attention_layernorm, post_attn_ln);
        if (!status) {
            return status;
        }
        status = require_f32(*backend_, layer.pre_feedforward_layernorm, pre_ffn_ln);
        if (!status) {
            return status;
        }
        status = require_f32(*backend_, layer.post_feedforward_layernorm, post_ffn_ln);
        if (!status) {
            return status;
        }
        status = require_f32(*backend_, layer.q_norm, q_norm);
        if (!status) {
            return status;
        }
        status = require_f32(*backend_, layer.k_norm, k_norm);
        if (!status) {
            return status;
        }
        status = require_f32(*backend_, layer.ple_norm, ple_post_norm);
        if (!status) {
            return status;
        }
        status = require_f32(*backend_, layer.layer_scalar, layer_scalar);
        if (!status) {
            return status;
        }

        std::memcpy(residual, hidden_states, static_cast<size_t>(seq * hidden) * sizeof(float));
        rms_norm(normed, hidden_states, input_ln, seq, hidden, eps);
        status = linear(q, normed, layer.q_proj, seq, hidden, q_dim);
        if (!status) {
            return status;
        }
        status = linear(k, normed, layer.k_proj, seq, hidden, kv_dim);
        if (!status) {
            return status;
        }
        status = linear(v, normed, layer.v_proj, seq, hidden, kv_dim);
        if (!status) {
            return status;
        }
        rms_norm(q, q, q_norm, seq * n_heads, head_dim, eps);
        rms_norm(k, k, k_norm, seq * n_kv, head_dim, eps);
        rms_norm(v, v, nullptr, seq * n_kv, head_dim, eps);
        apply_rope(q, seq, n_heads, head_dim, rope_inv_freq_[static_cast<size_t>(layer_index)].data());
        apply_rope(k, seq, n_kv, head_dim, rope_inv_freq_[static_cast<size_t>(layer_index)].data());
        attention(
            attn,
            q,
            k,
            v,
            attn_score,
            seq,
            n_heads,
            n_kv,
            head_dim,
            !spec.full_attention,
            static_cast<int64_t>(config_.sliding_window));
        status = linear(normed, attn, layer.o_proj, seq, q_dim, hidden);
        if (!status) {
            return status;
        }
        rms_norm(normed, normed, post_attn_ln, seq, hidden, eps);
        for (int64_t i = 0; i < seq * hidden; ++i) {
            hidden_states[i] = residual[i] + normed[i];
        }

        std::memcpy(residual, hidden_states, static_cast<size_t>(seq * hidden) * sizeof(float));
        rms_norm(normed, hidden_states, pre_ffn_ln, seq, hidden, eps);
        status = linear(gate, normed, layer.gate_proj, seq, hidden, inter);
        if (!status) {
            return status;
        }
        status = linear(up, normed, layer.up_proj, seq, hidden, inter);
        if (!status) {
            return status;
        }
        gelu_tanh(gate, seq * inter);
        for (int64_t i = 0; i < seq * inter; ++i) {
            gate[i] *= up[i];
        }
        status = linear(normed, gate, layer.down_proj, seq, inter, hidden);
        if (!status) {
            return status;
        }
        rms_norm(normed, normed, post_ffn_ln, seq, hidden, eps);
        for (int64_t i = 0; i < seq * hidden; ++i) {
            hidden_states[i] = residual[i] + normed[i];
        }

        std::memcpy(residual, hidden_states, static_cast<size_t>(seq * hidden) * sizeof(float));
        status = linear(ple_h, hidden_states, layer.ple_gate, seq, hidden, ple_dim);
        if (!status) {
            return status;
        }
        gelu_tanh(ple_h, seq * ple_dim);
        for (int64_t t = 0; t < seq; ++t) {
            const float* ple_row = ple + (t * n_layers + layer_index) * ple_dim;
            float* gate_row = ple_h + t * ple_dim;
            for (int64_t d = 0; d < ple_dim; ++d) {
                gate_row[d] *= ple_row[d];
            }
        }
        status = linear(normed, ple_h, layer.ple_proj, seq, ple_dim, hidden);
        if (!status) {
            return status;
        }
        rms_norm(normed, normed, ple_post_norm, seq, hidden, eps);
        const float scalar = layer_scalar[0];
        for (int64_t i = 0; i < seq * hidden; ++i) {
            hidden_states[i] = (residual[i] + normed[i]) * scalar;
        }
    }

    rms_norm(normed, hidden_states, final_norm, seq, hidden, eps);
    status = linear(projected, normed, model_.embedding_projection, seq, hidden, embed_dim);
    if (!status) {
        return status;
    }
    mean_pool(pooled, projected, seq, embed_dim);

    const int64_t width = static_cast<int64_t>(embedding.size());
    for (int64_t i = 0; i < width; ++i) {
        embedding[static_cast<size_t>(i)] = pooled[i];
    }
    l2_normalize(embedding.data(), width);
    if (!all_finite(embedding.data(), width)) {
        return Status::backend_error_status("embedding is not finite");
    }
    return Status::success();
}

}  // namespace tinyinfer
