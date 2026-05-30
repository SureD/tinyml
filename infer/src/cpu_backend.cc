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

    Status matmul_out(
        const TensorView& out,
        const TensorView& x,
        const TensorView& w) override {
        Status status = validate_f32_contiguous(out);
        if (!status) {
            return status;
        }
        status = validate_f32_contiguous(x);
        if (!status) {
            return status;
        }
        status = validate_f32_contiguous(w);
        if (!status) {
            return status;
        }
        if (x.shape.ndim != 2 || w.shape.ndim != 2) {
            return Status::invalid_argument_status("matmul expects x=[M,K] and w=[N,K]");
        }
        if (out.shape.ndim != 2 && out.shape.ndim != 3) {
            return Status::invalid_argument_status("matmul output expects rank 2 or rank 3");
        }

        const int64_t m = x.dim(0);
        const int64_t k = x.dim(1);
        const int64_t n = w.dim(0);
        if (m <= 0 || k <= 0 || n <= 0 || w.dim(1) != k) {
            return Status::invalid_argument_status("matmul shape mismatch");
        }
        if (out.dim(0) != m || out.numel() != m * n) {
            return Status::invalid_argument_status("matmul output shape mismatch");
        }

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
        return Status::success();
    }

    Status embedding_out(
        const TensorView& out,
        const TensorView& table,
        std::span<const uint32_t> token_ids) override {
        Status status = validate_f32_contiguous(out);
        if (!status) {
            return status;
        }
        status = validate_f32_contiguous(table);
        if (!status) {
            return status;
        }
        if (token_ids.empty()) {
            return Status::invalid_argument_status("embedding token_ids must not be empty");
        }
        if (out.shape.ndim != 2 || table.shape.ndim != 2) {
            return Status::invalid_argument_status("embedding expects out=[T,H], table=[V,H]");
        }
        if (out.dim(0) != static_cast<int64_t>(token_ids.size()) ||
            out.dim(1) != table.dim(1)) {
            return Status::invalid_argument_status("embedding output shape mismatch");
        }

        const int64_t vocab = table.dim(0);
        const int64_t hidden = table.dim(1);
        const float* table_data = f32_data(table);
        float* out_data = f32_data(out);
        for (size_t t = 0; t < token_ids.size(); ++t) {
            const uint32_t token = token_ids[t];
            if (static_cast<int64_t>(token) >= vocab) {
                return Status::invalid_argument_status("embedding token id exceeds vocab size");
            }
            const float* src = table_data + static_cast<int64_t>(token) * hidden;
            float* dst = out_data + static_cast<int64_t>(t) * hidden;
            std::memcpy(dst, src, static_cast<size_t>(hidden) * sizeof(float));
        }
        return Status::success();
    }

    Status add_inplace(
        const TensorView& dst,
        const TensorView& src) override {
        Status status = validate_f32_contiguous(dst);
        if (!status) {
            return status;
        }
        status = validate_f32_contiguous(src);
        if (!status) {
            return status;
        }
        if (!same_shape(dst, src)) {
            return Status::invalid_argument_status("add shape mismatch");
        }

        float* dst_data = f32_data(dst);
        const float* src_data = f32_data(src);
        const int64_t count = dst.numel();
        for (int64_t i = 0; i < count; ++i) {
            dst_data[i] += src_data[i];
        }
        return Status::success();
    }

    Status rms_norm_out(
        const TensorView& out,
        const TensorView& x,
        const TensorView& weight,
        float eps) override {
        Status status = validate_f32_contiguous(out);
        if (!status) {
            return status;
        }
        status = validate_f32_contiguous(x);
        if (!status) {
            return status;
        }
        status = validate_f32_contiguous(weight);
        if (!status) {
            return status;
        }
        if (eps <= 0.0f) {
            return Status::invalid_argument_status("RMSNorm eps must be positive");
        }
        if (x.shape.ndim != 2 || weight.shape.ndim != 1) {
            return Status::invalid_argument_status("RMSNorm expects x=[T,H] and weight=[H]");
        }
        if (!same_shape(out, x) || weight.dim(0) != x.dim(1)) {
            return Status::invalid_argument_status("RMSNorm shape mismatch");
        }

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
        return Status::success();
    }

    Status rope_inplace(
        const TensorView& q,
        const TensorView& k,
        uint32_t start_pos,
        float theta) override {
        Status status = validate_f32_contiguous(q);
        if (!status) {
            return status;
        }
        status = validate_f32_contiguous(k);
        if (!status) {
            return status;
        }
        if (theta <= 0.0f) {
            return Status::invalid_argument_status("RoPE theta must be positive");
        }
        if (q.shape.ndim != 3 || k.shape.ndim != 3) {
            return Status::invalid_argument_status("RoPE expects q=[T,H,D] and k=[T,KVH,D]");
        }
        if (q.dim(0) != k.dim(0) || q.dim(2) != k.dim(2)) {
            return Status::invalid_argument_status("RoPE q/k shape mismatch");
        }
        if ((q.dim(2) % 2) != 0) {
            return Status::invalid_argument_status("RoPE head dimension must be even");
        }

        apply_rope(f32_data(q), q.dim(0), q.dim(1), q.dim(2), start_pos, theta);
        apply_rope(f32_data(k), k.dim(0), k.dim(1), k.dim(2), start_pos, theta);
        return Status::success();
    }

    Status attention_out(
        const TensorView& out,
        const TensorView& q,
        const TensorView& k,
        const TensorView& v,
        const TensorView& k_cache,
        const TensorView& v_cache,
        uint32_t start_pos,
        uint32_t kv_len) override {
        Status status = validate_f32_contiguous(out);
        if (!status) {
            return status;
        }
        status = validate_f32_contiguous(q);
        if (!status) {
            return status;
        }
        status = validate_f32_contiguous(k);
        if (!status) {
            return status;
        }
        status = validate_f32_contiguous(v);
        if (!status) {
            return status;
        }
        status = validate_f32_contiguous(k_cache);
        if (!status) {
            return status;
        }
        status = validate_f32_contiguous(v_cache);
        if (!status) {
            return status;
        }
        if (q.shape.ndim != 3 || k.shape.ndim != 3 || v.shape.ndim != 3) {
            return Status::invalid_argument_status(
                "attention expects q=[T,H,D], k=[T,KVH,D], v=[T,KVH,D]");
        }
        if (k_cache.shape.ndim != 3 || v_cache.shape.ndim != 3) {
            return Status::invalid_argument_status(
                "attention cache expects [KVH,S,D] per layer");
        }
        if (out.shape.ndim != 2 && out.shape.ndim != 3) {
            return Status::invalid_argument_status("attention output expects rank 2 or rank 3");
        }

        const int64_t seq_len = q.dim(0);
        const int64_t n_heads = q.dim(1);
        const int64_t n_kv_heads = k.dim(1);
        const int64_t head_dim = q.dim(2);
        if (seq_len <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0) {
            return Status::invalid_argument_status("attention dimensions must be positive");
        }
        if (k.dim(0) != seq_len || v.dim(0) != seq_len ||
            v.dim(1) != n_kv_heads || k.dim(2) != head_dim || v.dim(2) != head_dim) {
            return Status::invalid_argument_status("attention q/k/v shape mismatch");
        }
        if (k_cache.dim(0) != n_kv_heads || v_cache.dim(0) != n_kv_heads ||
            k_cache.dim(2) != head_dim || v_cache.dim(2) != head_dim ||
            k_cache.dim(1) != v_cache.dim(1)) {
            return Status::invalid_argument_status("attention cache shape mismatch");
        }
        if ((n_heads % n_kv_heads) != 0) {
            return Status::invalid_argument_status("attention heads must be divisible by KV heads");
        }
        if (out.dim(0) != seq_len || out.numel() != seq_len * n_heads * head_dim) {
            return Status::invalid_argument_status("attention output shape mismatch");
        }
        if (kv_len == 0 || kv_len > static_cast<uint32_t>(k_cache.dim(1))) {
            return Status::invalid_argument_status("attention kv_len exceeds cache capacity");
        }
        if (start_pos > kv_len ||
            static_cast<uint32_t>(seq_len) > kv_len - start_pos) {
            return Status::invalid_argument_status("attention current tokens exceed kv_len");
        }

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
        return Status::success();
    }

    Status swiglu_out(
        const TensorView& out,
        const TensorView& gate,
        const TensorView& up) override {
        Status status = validate_f32_contiguous(out);
        if (!status) {
            return status;
        }
        status = validate_f32_contiguous(gate);
        if (!status) {
            return status;
        }
        status = validate_f32_contiguous(up);
        if (!status) {
            return status;
        }
        if (!same_shape(out, gate) || !same_shape(out, up)) {
            return Status::invalid_argument_status("SwiGLU shape mismatch");
        }

        const int64_t count = out.numel();
        float* out_data = f32_data(out);
        const float* gate_data = f32_data(gate);
        const float* up_data = f32_data(up);
        for (int64_t i = 0; i < count; ++i) {
            const float g = gate_data[i];
            out_data[i] = (g / (1.0f + std::exp(-g))) * up_data[i];
        }
        return Status::success();
    }

    Status argmax(
        uint32_t& out_token,
        const TensorView& logits) override {
        Status status = validate_f32_contiguous(logits);
        if (!status) {
            return status;
        }
        if (logits.shape.ndim != 1 &&
            !(logits.shape.ndim == 2 && logits.dim(0) == 1)) {
            return Status::invalid_argument_status("argmax expects logits shape [V] or [1,V]");
        }

        const int64_t count = logits.numel();
        if (count <= 0 ||
            static_cast<uint64_t>(count) > std::numeric_limits<uint32_t>::max()) {
            return Status::invalid_argument_status("argmax logits size is invalid");
        }

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
        return Status::success();
    }

    Status synchronize() override {
        return Status::success();
    }

protected:
    Status release_arena(MemoryArena& arena) override {
        void* handle = arena_handle(arena);
        if (handle != nullptr) {
            ::operator delete(handle, std::align_val_t(kArenaAlignment));
        }
        return Status::success();
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

    Status validate_f32_contiguous(const TensorView& view) const {
        Status status = validate_cpu_contiguous(view);
        if (!status) {
            return status;
        }
        if (view.dtype != DType::f32) {
            return Status::unimplemented_status("CPU math only supports f32 tensors");
        }
        return Status::success();
    }

    bool same_shape(const TensorView& a, const TensorView& b) const {
        if (a.shape.ndim != b.shape.ndim) {
            return false;
        }
        for (uint32_t i = 0; i < a.shape.ndim; ++i) {
            if (a.shape.dims[i] != b.shape.dims[i]) {
                return false;
            }
        }
        return true;
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
