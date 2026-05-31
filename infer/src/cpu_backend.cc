#include "tinyinfer/backend.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <new>

namespace tinyinfer {
namespace {

constexpr size_t kArenaAlignment = 64;

class CPUBackend final : public Backend {
public:
    Device device() const override {
        return {DeviceType::cpu, 0};
    }

    Status alloc_arena(
        MemoryArena& arena,
        size_t bytes,
        MemoryKind kind) override {
        if (bytes == 0) {
            return Status::invalid_argument_status("CPU arena size must be non-zero");
        }

        void* handle = ::operator new(
            bytes,
            std::align_val_t(kArenaAlignment),
            std::nothrow);
        if (handle == nullptr) {
            return Status::backend_error_status("CPU arena allocation failed");
        }

        bind_arena(arena, handle, bytes, kind);
        return Status::success();
    }

    Status copy_from_host(
        const TensorView& dst,
        const void* src,
        size_t bytes) override {
        Status status = validate_copy_view(dst, bytes);
        if (!status) {
            return status;
        }
        if (bytes == 0) {
            return Status::success();
        }
        if (src == nullptr) {
            return Status::invalid_argument_status("source host pointer is null");
        }

        std::memcpy(data(dst), src, bytes);
        return Status::success();
    }

    Status copy_to_host(
        void* dst,
        const TensorView& src,
        size_t bytes) override {
        Status status = validate_copy_view(src, bytes);
        if (!status) {
            return status;
        }
        if (bytes == 0) {
            return Status::success();
        }
        if (dst == nullptr) {
            return Status::invalid_argument_status("destination host pointer is null");
        }

        std::memcpy(dst, data(src), bytes);
        return Status::success();
    }

    void matmul_out(
        const TensorView& out,
        const TensorView& x,
        const TensorView& w) override {
        const int64_t m = x.dim(0);
        const int64_t k = x.dim(1);
        const int64_t n = w.dim(0);
        const float* x_data = f32_data(x);
        const float* w_data = f32_data(w);
        float* out_data = f32_data(out);
        for (int64_t row = 0; row < m; ++row) {
            for (int64_t col = 0; col < n; ++col) {
                float sum = 0.0f;
                for (int64_t inner = 0; inner < k; ++inner) {
                    sum += x_data[row * k + inner] * w_data[col * k + inner];
                }
                out_data[row * n + col] = sum;
            }
        }
    }

    void embedding_out(
        const TensorView& out,
        const TensorView& table,
        std::span<const uint32_t> token_ids) override {
        const int64_t hidden = table.dim(1);
        const float* table_data = f32_data(table);
        float* out_data = f32_data(out);
        for (size_t t = 0; t < token_ids.size(); ++t) {
            const uint32_t token = token_ids[t];
            const float* src = table_data + static_cast<int64_t>(token) * hidden;
            float* dst = out_data + static_cast<int64_t>(t) * hidden;
            std::memcpy(dst, src, static_cast<size_t>(hidden) * sizeof(float));
        }
    }

    void add_inplace(
        const TensorView& dst,
        const TensorView& src) override {
        float* dst_data = f32_data(dst);
        const float* src_data = f32_data(src);
        const int64_t count = dst.numel();
        for (int64_t i = 0; i < count; ++i) {
            dst_data[i] += src_data[i];
        }
    }

    void rms_norm_out(
        const TensorView& out,
        const TensorView& x,
        const TensorView& weight,
        float eps) override {
        const int64_t rows = x.dim(0);
        const int64_t hidden = x.dim(1);
        const float* x_data = f32_data(x);
        const float* weight_data = f32_data(weight);
        float* out_data = f32_data(out);
        for (int64_t row = 0; row < rows; ++row) {
            const float* x_row = x_data + row * hidden;
            float* out_row = out_data + row * hidden;

            float sum_sq = 0.0f;
            for (int64_t i = 0; i < hidden; ++i) {
                sum_sq += x_row[i] * x_row[i];
            }
            const float mean_sq = sum_sq / static_cast<float>(hidden);
            const float scale = 1.0f / std::sqrt(mean_sq + eps);
            for (int64_t i = 0; i < hidden; ++i) {
                out_row[i] = x_row[i] * scale * weight_data[i];
            }
        }
    }

    void rope_inplace(
        const TensorView& q,
        const TensorView& k,
        uint32_t start_pos,
        float theta) override {
        apply_rope(f32_data(q), q.dim(0), q.dim(1), q.dim(2), start_pos, theta);
        apply_rope(f32_data(k), k.dim(0), k.dim(1), k.dim(2), start_pos, theta);
    }

    void attention_out(
        const TensorView& out,
        const TensorView& q,
        const TensorView& k,
        const TensorView& v,
        const TensorView& k_cache,
        const TensorView& v_cache,
        uint32_t start_pos,
        uint32_t kv_len) override {
        const int64_t seq_len = q.dim(0);
        const int64_t n_heads = q.dim(1);
        const int64_t n_kv_heads = k.dim(1);
        const int64_t head_dim = q.dim(2);

        write_kv_cache(
            f32_data(k_cache),
            f32_data(v_cache),
            f32_data(k),
            f32_data(v),
            seq_len,
            n_kv_heads,
            k_cache.dim(1),
            head_dim,
            start_pos);
        compute_attention(
            f32_data(out),
            f32_data(q),
            f32_data(k_cache),
            f32_data(v_cache),
            seq_len,
            n_heads,
            n_kv_heads,
            k_cache.dim(1),
            head_dim,
            start_pos,
            kv_len);
    }

    void swiglu_out(
        const TensorView& out,
        const TensorView& gate,
        const TensorView& up) override {
        const int64_t count = out.numel();
        float* out_data = f32_data(out);
        const float* gate_data = f32_data(gate);
        const float* up_data = f32_data(up);
        for (int64_t i = 0; i < count; ++i) {
            const float g = gate_data[i];
            out_data[i] = (g / (1.0f + std::exp(-g))) * up_data[i];
        }
    }

    void argmax(
        uint32_t& out_token,
        const TensorView& logits) override {
        const int64_t count = logits.numel();
        const float* values = f32_data(logits);
        uint32_t best_index = 0;
        float best_value = values[0];
        for (uint32_t i = 1; i < static_cast<uint32_t>(count); ++i) {
            if (values[i] > best_value) {
                best_value = values[i];
                best_index = i;
            }
        }

        out_token = best_index;
    }

    Status synchronize() override {
        return Status::success();
    }

protected:
    void release_arena(MemoryArena& arena) noexcept override {
        void* handle = arena_handle(arena);
        if (handle != nullptr) {
            ::operator delete(handle, std::align_val_t(kArenaAlignment));
        }
    }

private:
    Status validate_copy_view(const TensorView& view, size_t bytes) const {
        Status status = validate_cpu_contiguous(view);
        if (!status) {
            return status;
        }
        if (bytes > view.logical_nbytes()) {
            return Status::invalid_argument_status("copy exceeds tensor logical byte size");
        }
        return Status::success();
    }

    Status validate_cpu_contiguous(const TensorView& view) const {
        if (!view.defined()) {
            return Status::invalid_argument_status("tensor view is not defined");
        }
        if (!owns_arena(*view.arena)) {
            return Status::invalid_argument_status("tensor view belongs to a different backend");
        }
        if (view.device().type != DeviceType::cpu) {
            return Status::invalid_argument_status("tensor view is not on CPU");
        }
        if (!view.contiguous()) {
            return Status::invalid_argument_status("tensor view must be contiguous");
        }
        return Status::success();
    }

    void apply_rope(
        float* values,
        int64_t seq_len,
        int64_t heads,
        int64_t head_dim,
        uint32_t start_pos,
        float theta) {
        const int64_t half = head_dim / 2;
        for (int64_t t = 0; t < seq_len; ++t) {
            const float pos = static_cast<float>(start_pos) + static_cast<float>(t);
            for (int64_t h = 0; h < heads; ++h) {
                float* row = values + (t * heads + h) * head_dim;
                for (int64_t i = 0; i < half; ++i) {
                    const float exponent =
                        static_cast<float>(2 * i) / static_cast<float>(head_dim);
                    const float angle = pos / std::pow(theta, exponent);
                    const float c = std::cos(angle);
                    const float s = std::sin(angle);
                    const float x0 = row[i];
                    const float x1 = row[i + half];
                    row[i] = x0 * c - x1 * s;
                    row[i + half] = x1 * c + x0 * s;
                }
            }
        }
    }

    void write_kv_cache(
        float* k_cache,
        float* v_cache,
        const float* k,
        const float* v,
        int64_t seq_len,
        int64_t n_kv_heads,
        int64_t max_seq_len,
        int64_t head_dim,
        uint32_t start_pos) {
        // Source K/V are contiguous [seq_len, n_kv_heads, head_dim].
        const int64_t src_token_stride = n_kv_heads * head_dim;
        const int64_t src_head_stride = head_dim;
        // Cache K/V are contiguous [n_kv_heads, max_seq_len, head_dim].
        const int64_t cache_head_stride = max_seq_len * head_dim;
        const int64_t cache_pos_stride = head_dim;

        for (int64_t t = 0; t < seq_len; ++t) {
            const int64_t pos = static_cast<int64_t>(start_pos) + t;
            for (int64_t h = 0; h < n_kv_heads; ++h) {
                float* k_dst =
                    k_cache + h * cache_head_stride + pos * cache_pos_stride;
                float* v_dst =
                    v_cache + h * cache_head_stride + pos * cache_pos_stride;
                const float* k_src = k + t * src_token_stride + h * src_head_stride;
                const float* v_src = v + t * src_token_stride + h * src_head_stride;
                std::memcpy(k_dst, k_src, static_cast<size_t>(head_dim) * sizeof(float));
                std::memcpy(v_dst, v_src, static_cast<size_t>(head_dim) * sizeof(float));
            }
        }
    }

    float attention_score(
        const float* q_row,
        const float* k_cache,
        int64_t kv_head,
        int64_t key_pos,
        int64_t cache_head_stride,
        int64_t cache_pos_stride,
        int64_t head_dim,
        float scale) const {
        const float* k_row =
            k_cache + kv_head * cache_head_stride + key_pos * cache_pos_stride;
        float dot = 0.0f;
        for (int64_t d = 0; d < head_dim; ++d) {
            dot += q_row[d] * k_row[d];
        }
        return dot * scale;
    }

    void compute_attention(
        float* out,
        const float* q,
        const float* k_cache,
        const float* v_cache,
        int64_t seq_len,
        int64_t n_heads,
        int64_t n_kv_heads,
        int64_t max_seq_len,
        int64_t head_dim,
        uint32_t start_pos,
        uint32_t kv_len) const {
        const int64_t group_size = n_heads / n_kv_heads;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        // Q/out are contiguous [seq_len, n_heads, head_dim].
        const int64_t q_token_stride = n_heads * head_dim;
        const int64_t q_head_stride = head_dim;
        const int64_t out_token_stride = n_heads * head_dim;
        const int64_t out_head_stride = head_dim;
        // K/V cache are contiguous [n_kv_heads, max_seq_len, head_dim].
        const int64_t cache_head_stride = max_seq_len * head_dim;
        const int64_t cache_pos_stride = head_dim;

        for (int64_t t = 0; t < seq_len; ++t) {
            const int64_t query_pos = static_cast<int64_t>(start_pos) + t;
            const int64_t valid_len =
                std::min(static_cast<int64_t>(kv_len), query_pos + 1);
            for (int64_t h = 0; h < n_heads; ++h) {
                const int64_t kv_head = h / group_size;
                const float* q_row = q + t * q_token_stride + h * q_head_stride;
                float* out_row = out + t * out_token_stride + h * out_head_stride;

                float max_score = -std::numeric_limits<float>::infinity();
                for (int64_t p = 0; p < valid_len; ++p) {
                    const float score = attention_score(
                        q_row,
                        k_cache,
                        kv_head,
                        p,
                        cache_head_stride,
                        cache_pos_stride,
                        head_dim,
                        scale);
                    if (score > max_score) {
                        max_score = score;
                    }
                }

                float denom = 0.0f;
                for (int64_t p = 0; p < valid_len; ++p) {
                    const float score = attention_score(
                        q_row,
                        k_cache,
                        kv_head,
                        p,
                        cache_head_stride,
                        cache_pos_stride,
                        head_dim,
                        scale);
                    denom += std::exp(score - max_score);
                }

                for (int64_t d = 0; d < head_dim; ++d) {
                    float sum = 0.0f;
                    for (int64_t p = 0; p < valid_len; ++p) {
                        const float score = attention_score(
                            q_row,
                            k_cache,
                            kv_head,
                            p,
                            cache_head_stride,
                            cache_pos_stride,
                            head_dim,
                            scale);
                        const float weight = std::exp(score - max_score) / denom;
                        const float* v_row =
                            v_cache + kv_head * cache_head_stride
                            + p * cache_pos_stride;
                        sum += weight * v_row[d];
                    }
                    out_row[d] = sum;
                }
            }
        }
    }

    uint8_t* data(const TensorView& view) {
        return static_cast<uint8_t*>(arena_handle(*view.arena)) + view.byte_offset;
    }

    const uint8_t* data(const TensorView& view) const {
        return static_cast<const uint8_t*>(arena_handle(*view.arena)) + view.byte_offset;
    }

    float* f32_data(const TensorView& view) {
        return reinterpret_cast<float*>(data(view));
    }

    const float* f32_data(const TensorView& view) const {
        return reinterpret_cast<const float*>(data(view));
    }
};

}  // namespace

Result<std::unique_ptr<Backend>> create_cpu_backend() {
    return {Status::success(), std::make_unique<CPUBackend>()};
}

}  // namespace tinyinfer
