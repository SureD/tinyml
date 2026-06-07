#include "tinyinfer/backend.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#if defined(__APPLE__) && __has_include(<Metal/Metal.h>)
#import <Metal/Metal.h>
#define TINYINFER_HAS_METAL 1
#else
#define TINYINFER_HAS_METAL 0
#endif

namespace tinyinfer {
namespace {

#if TINYINFER_HAS_METAL

constexpr const char* kMetalSource = R"metal(
#include <metal_stdlib>
using namespace metal;

struct MatmulParams {
    uint m;
    uint k;
    uint n;
};

struct EmbeddingParams {
    uint token_count;
    uint hidden;
};

struct AddParams {
    uint count;
};

struct RmsNormParams {
    uint rows;
    uint hidden;
    float eps;
};

struct RopeParams {
    uint seq_len;
    uint heads;
    uint head_dim;
    uint start_pos;
    float theta;
};

struct WriteKVParams {
    uint seq_len;
    uint n_kv_heads;
    uint max_seq_len;
    uint head_dim;
    uint start_pos;
};

struct AttentionParams {
    uint seq_len;
    uint n_heads;
    uint n_kv_heads;
    uint max_seq_len;
    uint head_dim;
    uint start_pos;
    uint kv_len;
};

struct UnaryParams {
    uint count;
};

struct ArgmaxParams {
    uint count;
};

struct ArgmaxPair {
    float value;
    uint index;
};

kernel void matmul_f32(
    device float* out [[buffer(0)]],
    const device float* x [[buffer(1)]],
    const device float* w [[buffer(2)]],
    constant MatmulParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    const uint total = p.m * p.n;
    if (gid >= total) {
        return;
    }

    const uint row = gid / p.n;
    const uint col = gid - row * p.n;
    float sum = 0.0f;
    for (uint inner = 0; inner < p.k; ++inner) {
        sum += x[row * p.k + inner] * w[col * p.k + inner];
    }
    out[gid] = sum;
}

kernel void matvec_f32(
    device float* out [[buffer(0)]],
    const device float* x [[buffer(1)]],
    const device float* w [[buffer(2)]],
    constant MatmulParams& p [[buffer(3)]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint lane_id [[thread_index_in_simdgroup]],
    uint output_in_threadgroup [[simdgroup_index_in_threadgroup]]) {
    constexpr uint simd_width = 32;
    constexpr uint outputs_per_threadgroup = 4;

    // Split the N outputs across SIMD groups. Each SIMD group computes one output.
    const uint output_index =
        threadgroup_id * outputs_per_threadgroup + output_in_threadgroup;
    if (output_index >= p.n) {
        return;
    }

    // The 32 lanes cooperate on the K dimension of this output's dot product.
    float partial_sum = 0.0f;
    if ((p.k & 3u) == 0) {
        const uint k_vector_count = p.k / 4;
        const device float4* x_vectors = (const device float4*)x;
        const device float4* weight_row =
            (const device float4*)(w + output_index * p.k);
        for (uint k_vector = lane_id;
             k_vector < k_vector_count;
             k_vector += simd_width) {
            partial_sum += dot(x_vectors[k_vector], weight_row[k_vector]);
        }
    } else {
        const device float* weight_row = w + output_index * p.k;
        for (uint k = lane_id; k < p.k; k += simd_width) {
            partial_sum += x[k] * weight_row[k];
        }
    }

    const float output_value = simd_sum(partial_sum);
    if (lane_id == 0) {
        out[output_index] = output_value;
    }
}

kernel void small_m_gemm_f32(
    device float* out [[buffer(0)]],
    const device float* x [[buffer(1)]],
    const device float* w [[buffer(2)]],
    constant MatmulParams& p [[buffer(3)]],
    uint2 threadgroup_id [[threadgroup_position_in_grid]],
    uint lane_id [[thread_index_in_simdgroup]],
    uint output_col_in_tile [[simdgroup_index_in_threadgroup]]) {
    constexpr uint simd_width = 32;
    constexpr uint output_rows_per_tile = 4;
    constexpr uint output_cols_per_tile = 4;

    // 1. One threadgroup owns a 4x4 tile in the [M, N] output matrix.
    const uint output_row_begin = threadgroup_id.y * output_rows_per_tile;
    const uint output_col =
        threadgroup_id.x * output_cols_per_tile + output_col_in_tile;
    if (output_col >= p.n) {
        return;
    }

    // 2. One SIMD group computes one column of the tile and reuses each
    // weight value across up to four output rows.
    float partial_sums[output_rows_per_tile] = {};
    if ((p.k & 3u) == 0) {
        const uint k_vector_count = p.k / 4;
        const device float4* weight_row =
            (const device float4*)(w + output_col * p.k);

        // 3. The 32 lanes split K. Each lane accumulates its part of four dots.
        for (uint k_vector = lane_id;
             k_vector < k_vector_count;
             k_vector += simd_width) {
            const float4 weight_values = weight_row[k_vector];
            for (uint row_in_tile = 0;
                 row_in_tile < output_rows_per_tile;
                 ++row_in_tile) {
                const uint output_row = output_row_begin + row_in_tile;
                if (output_row < p.m) {
                    const device float4* x_row =
                        (const device float4*)(x + output_row * p.k);
                    partial_sums[row_in_tile] +=
                        dot(x_row[k_vector], weight_values);
                }
            }
        }
    } else {
        const device float* weight_row = w + output_col * p.k;
        for (uint k = lane_id; k < p.k; k += simd_width) {
            const float weight_value = weight_row[k];
            for (uint row_in_tile = 0;
                 row_in_tile < output_rows_per_tile;
                 ++row_in_tile) {
                const uint output_row = output_row_begin + row_in_tile;
                if (output_row < p.m) {
                    partial_sums[row_in_tile] +=
                        x[output_row * p.k + k] * weight_value;
                }
            }
        }
    }

    for (uint row_in_tile = 0;
         row_in_tile < output_rows_per_tile;
         ++row_in_tile) {
        const float output_value = simd_sum(partial_sums[row_in_tile]);
        const uint output_row = output_row_begin + row_in_tile;
        if (lane_id == 0 && output_row < p.m) {
            out[output_row * p.n + output_col] = output_value;
        }
    }
}

kernel void small_m_gemm_tiled_f32(
    device float* out [[buffer(0)]],
    const device float* x [[buffer(1)]],
    const device float* w [[buffer(2)]],
    constant MatmulParams& p [[buffer(3)]],
    uint2 threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]],
    uint output_col_in_tile [[simdgroup_index_in_threadgroup]]) {
    constexpr uint simd_width = 32;
    constexpr uint output_rows_per_tile = 4;
    constexpr uint output_cols_per_tile = 4;
    constexpr uint k_per_tile = 128;
    constexpr uint k_vectors_per_tile = k_per_tile / 4;

    // 1. One threadgroup owns a 4x4 output tile.
    const uint output_row_begin = threadgroup_id.y * output_rows_per_tile;
    const uint output_col =
        threadgroup_id.x * output_cols_per_tile + output_col_in_tile;
    const bool valid_output_col = output_col < p.n;

    // 2. All 128 threads cooperatively load one X[4, 128] tile.
    // Each SIMD group then computes one output column from the shared X tile.
    threadgroup float4 x_tile[output_rows_per_tile * k_vectors_per_tile];
    const uint x_row_in_tile = thread_id / k_vectors_per_tile;
    const uint x_k_vector = thread_id % k_vectors_per_tile;

    float partial_sums[output_rows_per_tile] = {};
    for (uint k_begin = 0; k_begin < p.k; k_begin += k_per_tile) {
        const uint output_row = output_row_begin + x_row_in_tile;
        float4 x_values = float4(0.0f);
        if (output_row < p.m) {
            const device float4* x_row =
                (const device float4*)(x + output_row * p.k + k_begin);
            x_values = x_row[x_k_vector];
        }
        x_tile[thread_id] = x_values;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. The 32 lanes split BK=128 as 32 float4 vectors. A loaded
        // weight vector is reused across all four M rows in this tile.
        if (valid_output_col) {
            const device float4* weight_row =
                (const device float4*)(w + output_col * p.k + k_begin);
            const float4 weight_values = weight_row[lane_id];
            for (uint row_in_tile = 0;
                 row_in_tile < output_rows_per_tile;
                 ++row_in_tile) {
                partial_sums[row_in_tile] += dot(
                    x_tile[row_in_tile * k_vectors_per_tile + lane_id],
                    weight_values);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint row_in_tile = 0;
         row_in_tile < output_rows_per_tile;
         ++row_in_tile) {
        const float output_value = simd_sum(partial_sums[row_in_tile]);
        const uint output_row = output_row_begin + row_in_tile;
        if (lane_id == 0 && valid_output_col && output_row < p.m) {
            out[output_row * p.n + output_col] = output_value;
        }
    }
}

kernel void embedding_f32(
    device float* out [[buffer(0)]],
    const device float* table [[buffer(1)]],
    const device uint* token_ids [[buffer(2)]],
    constant EmbeddingParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    const uint total = p.token_count * p.hidden;
    if (gid >= total) {
        return;
    }

    const uint token_index = gid / p.hidden;
    const uint hidden_index = gid - token_index * p.hidden;
    const uint token = token_ids[token_index];
    out[gid] = table[token * p.hidden + hidden_index];
}

kernel void add_inplace_f32(
    device float* dst [[buffer(0)]],
    const device float* src [[buffer(1)]],
    constant AddParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.count) {
        return;
    }
    dst[gid] += src[gid];
}

kernel void rms_norm_f32(
    device float* out [[buffer(0)]],
    const device float* x [[buffer(1)]],
    const device float* weight [[buffer(2)]],
    constant RmsNormParams& p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
    constexpr uint simd_width = 32;
    constexpr uint threads_per_threadgroup = 256;
    constexpr uint simdgroups_per_threadgroup =
        threads_per_threadgroup / simd_width;

    if (row >= p.rows) {
        return;
    }

    const uint base = row * p.hidden;
    float partial_sum_sq = 0.0f;
    for (uint i = thread_id; i < p.hidden; i += threads_per_threadgroup) {
        const float value = x[base + i];
        partial_sum_sq += value * value;
    }

    partial_sum_sq = simd_sum(partial_sum_sq);
    threadgroup float simd_sums[simdgroups_per_threadgroup];
    if (lane_id == 0) {
        simd_sums[simdgroup_id] = partial_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_id == 0) {
        float sum_sq =
            lane_id < simdgroups_per_threadgroup ? simd_sums[lane_id] : 0.0f;
        sum_sq = simd_sum(sum_sq);
        if (lane_id == 0) {
            const float mean_sq = sum_sq / float(p.hidden);
            simd_sums[0] = rsqrt(mean_sq + p.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float scale = simd_sums[0];
    for (uint i = thread_id; i < p.hidden; i += threads_per_threadgroup) {
        out[base + i] = x[base + i] * scale * weight[i];
    }
}

kernel void add_rms_norm_f32(
    device float* norm_out [[buffer(0)]],
    device float* residual_out [[buffer(1)]],
    const device float* residual [[buffer(2)]],
    const device float* weight [[buffer(3)]],
    constant RmsNormParams& p [[buffer(4)]],
    threadgroup float* row_values [[threadgroup(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
    constexpr uint simd_width = 32;
    constexpr uint threads_per_threadgroup = 256;
    constexpr uint simdgroups_per_threadgroup =
        threads_per_threadgroup / simd_width;

    if (row >= p.rows) {
        return;
    }

    const uint base = row * p.hidden;
    float partial_sum_sq = 0.0f;
    for (uint i = thread_id; i < p.hidden; i += threads_per_threadgroup) {
        const float value = residual_out[base + i] + residual[base + i];
        residual_out[base + i] = value;
        row_values[i] = value;
        partial_sum_sq += value * value;
    }

    partial_sum_sq = simd_sum(partial_sum_sq);
    threadgroup float simd_sums[simdgroups_per_threadgroup];
    if (lane_id == 0) {
        simd_sums[simdgroup_id] = partial_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_id == 0) {
        float sum_sq =
            lane_id < simdgroups_per_threadgroup ? simd_sums[lane_id] : 0.0f;
        sum_sq = simd_sum(sum_sq);
        if (lane_id == 0) {
            const float mean_sq = sum_sq / float(p.hidden);
            simd_sums[0] = rsqrt(mean_sq + p.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float scale = simd_sums[0];
    for (uint i = thread_id; i < p.hidden; i += threads_per_threadgroup) {
        norm_out[base + i] = row_values[i] * scale * weight[i];
    }
}

kernel void rope_f32(
    device float* values [[buffer(0)]],
    constant RopeParams& p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const uint half_dim = p.head_dim / 2;
    const uint total = p.seq_len * p.heads * half_dim;
    if (gid >= total) {
        return;
    }

    const uint i = gid % half_dim;
    const uint head = (gid / half_dim) % p.heads;
    const uint token = gid / (half_dim * p.heads);
    const uint row_base = (token * p.heads + head) * p.head_dim;

    const float pos = float(p.start_pos + token);
    const float exponent = float(2 * i) / float(p.head_dim);
    const float angle = pos / pow(p.theta, exponent);
    const float c = cos(angle);
    const float s = sin(angle);
    const float x0 = values[row_base + i];
    const float x1 = values[row_base + i + half_dim];
    values[row_base + i] = x0 * c - x1 * s;
    values[row_base + i + half_dim] = x1 * c + x0 * s;
}

kernel void write_kv_cache_f32(
    device float* k_cache [[buffer(0)]],
    device float* v_cache [[buffer(1)]],
    const device float* k [[buffer(2)]],
    const device float* v [[buffer(3)]],
    constant WriteKVParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    const uint total = p.seq_len * p.n_kv_heads * p.head_dim;
    if (gid >= total) {
        return;
    }

    const uint d = gid % p.head_dim;
    const uint h = (gid / p.head_dim) % p.n_kv_heads;
    const uint token = gid / (p.head_dim * p.n_kv_heads);
    const uint src = token * p.n_kv_heads * p.head_dim + h * p.head_dim + d;
    const uint dst = h * p.max_seq_len * p.head_dim
        + (p.start_pos + token) * p.head_dim
        + d;
    k_cache[dst] = k[src];
    v_cache[dst] = v[src];
}

static inline float attention_score(
    const device float* q,
    const device float* k_cache,
    uint token,
    uint head,
    uint kv_head,
    uint key_pos,
    constant AttentionParams& p) {
    const uint q_base = token * p.n_heads * p.head_dim + head * p.head_dim;
    const uint k_base = kv_head * p.max_seq_len * p.head_dim + key_pos * p.head_dim;
    float dot = 0.0f;
    for (uint d = 0; d < p.head_dim; ++d) {
        dot += q[q_base + d] * k_cache[k_base + d];
    }
    return dot * rsqrt(float(p.head_dim));
}

kernel void attention_f32(
    device float* out [[buffer(0)]],
    const device float* q [[buffer(1)]],
    const device float* k_cache [[buffer(2)]],
    const device float* v_cache [[buffer(3)]],
    constant AttentionParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    const uint total = p.seq_len * p.n_heads * p.head_dim;
    if (gid >= total) {
        return;
    }

    const uint d = gid % p.head_dim;
    const uint head = (gid / p.head_dim) % p.n_heads;
    const uint token = gid / (p.head_dim * p.n_heads);
    const uint group_size = p.n_heads / p.n_kv_heads;
    const uint kv_head = head / group_size;
    const uint query_pos = p.start_pos + token;
    const uint valid_len = min(p.kv_len, query_pos + 1);

    float max_score = -3.402823466e+38f;
    for (uint key_pos = 0; key_pos < valid_len; ++key_pos) {
        const float score = attention_score(q, k_cache, token, head, kv_head, key_pos, p);
        max_score = max(max_score, score);
    }

    float denom = 0.0f;
    for (uint key_pos = 0; key_pos < valid_len; ++key_pos) {
        const float score = attention_score(q, k_cache, token, head, kv_head, key_pos, p);
        denom += exp(score - max_score);
    }

    float sum = 0.0f;
    for (uint key_pos = 0; key_pos < valid_len; ++key_pos) {
        const float score = attention_score(q, k_cache, token, head, kv_head, key_pos, p);
        const float weight = exp(score - max_score) / denom;
        const uint v_index = kv_head * p.max_seq_len * p.head_dim
            + key_pos * p.head_dim
            + d;
        sum += weight * v_cache[v_index];
    }

    out[gid] = sum;
}

kernel void swiglu_f32(
    device float* out [[buffer(0)]],
    const device float* gate [[buffer(1)]],
    const device float* up [[buffer(2)]],
    constant UnaryParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.count) {
        return;
    }

    const float g = gate[gid];
    out[gid] = (g / (1.0f + exp(-g))) * up[gid];
}

static inline ArgmaxPair better_argmax(ArgmaxPair lhs, ArgmaxPair rhs) {
    if (rhs.value > lhs.value ||
        (rhs.value == lhs.value && rhs.index < lhs.index)) {
        return rhs;
    }
    return lhs;
}

kernel void matvec_argmax_partial_f32(
    device ArgmaxPair* partials [[buffer(0)]],
    const device float* x [[buffer(1)]],
    const device float* w [[buffer(2)]],
    constant MatmulParams& p [[buffer(3)]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]],
    uint output_in_threadgroup [[simdgroup_index_in_threadgroup]]) {
    constexpr uint simd_width = 32;
    constexpr uint outputs_per_threadgroup = 4;

    const uint output_index =
        threadgroup_id * outputs_per_threadgroup + output_in_threadgroup;
    float partial_sum = 0.0f;
    if (output_index < p.n) {
        if ((p.k & 3u) == 0) {
            const uint k_vector_count = p.k / 4;
            const device float4* x_vectors = (const device float4*)x;
            const device float4* weight_row =
                (const device float4*)(w + output_index * p.k);
            for (uint k_vector = lane_id;
                 k_vector < k_vector_count;
                 k_vector += simd_width) {
                partial_sum += dot(x_vectors[k_vector], weight_row[k_vector]);
            }
        } else {
            const device float* weight_row = w + output_index * p.k;
            for (uint k = lane_id; k < p.k; k += simd_width) {
                partial_sum += x[k] * weight_row[k];
            }
        }
    }

    const float output_value = simd_sum(partial_sum);
    threadgroup ArgmaxPair output_candidates[outputs_per_threadgroup];
    if (lane_id == 0) {
        output_candidates[output_in_threadgroup] = {
            output_index < p.n ? output_value : -3.402823466e+38f,
            output_index < p.n ? output_index : 0xffffffffu,
        };
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (thread_id == 0) {
        ArgmaxPair best = output_candidates[0];
        for (uint i = 1; i < outputs_per_threadgroup; ++i) {
            best = better_argmax(best, output_candidates[i]);
        }
        partials[threadgroup_id] = best;
    }
}

kernel void argmax_reduce_pairs_f32(
    device uint* out_index [[buffer(0)]],
    const device ArgmaxPair* partials [[buffer(1)]],
    constant ArgmaxParams& p [[buffer(2)]],
    uint thread_id [[thread_index_in_threadgroup]]) {
    constexpr uint threads_per_threadgroup = 256;
    ArgmaxPair best = {-3.402823466e+38f, 0xffffffffu};
    for (uint i = thread_id; i < p.count; i += threads_per_threadgroup) {
        best = better_argmax(best, partials[i]);
    }

    threadgroup ArgmaxPair candidates[threads_per_threadgroup];
    candidates[thread_id] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_threadgroup / 2;
         stride != 0;
         stride /= 2) {
        if (thread_id < stride) {
            candidates[thread_id] = better_argmax(
                candidates[thread_id],
                candidates[thread_id + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (thread_id == 0) {
        out_index[0] = candidates[0].index;
    }
}

kernel void argmax_f32(
    device uint* out_token [[buffer(0)]],
    const device float* logits [[buffer(1)]],
    constant ArgmaxParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid != 0 || p.count == 0) {
        return;
    }

    uint best_index = 0;
    float best_value = logits[0];
    for (uint i = 1; i < p.count; ++i) {
        const float value = logits[i];
        if (value > best_value) {
            best_value = value;
            best_index = i;
        }
    }
    out_token[0] = best_index;
}
)metal";

struct MatmulParams {
    uint32_t m;
    uint32_t k;
    uint32_t n;
};

struct EmbeddingParams {
    uint32_t token_count;
    uint32_t hidden;
};

struct AddParams {
    uint32_t count;
};

struct RmsNormParams {
    uint32_t rows;
    uint32_t hidden;
    float eps;
};

struct RopeParams {
    uint32_t seq_len;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t start_pos;
    float theta;
};

struct WriteKVParams {
    uint32_t seq_len;
    uint32_t n_kv_heads;
    uint32_t max_seq_len;
    uint32_t head_dim;
    uint32_t start_pos;
};

struct AttentionParams {
    uint32_t seq_len;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t max_seq_len;
    uint32_t head_dim;
    uint32_t start_pos;
    uint32_t kv_len;
};

struct UnaryParams {
    uint32_t count;
};

struct ArgmaxParams {
    uint32_t count;
};

struct ArgmaxPair {
    float value;
    uint32_t index;
};

struct Pipelines {
    id<MTLComputePipelineState> matmul = nil;
    id<MTLComputePipelineState> matvec = nil;
    id<MTLComputePipelineState> small_m_gemm = nil;
    id<MTLComputePipelineState> small_m_gemm_tiled = nil;
    id<MTLComputePipelineState> embedding = nil;
    id<MTLComputePipelineState> add = nil;
    id<MTLComputePipelineState> add_rms_norm = nil;
    id<MTLComputePipelineState> rms_norm = nil;
    id<MTLComputePipelineState> rope = nil;
    id<MTLComputePipelineState> write_kv_cache = nil;
    id<MTLComputePipelineState> attention = nil;
    id<MTLComputePipelineState> swiglu = nil;
    id<MTLComputePipelineState> matvec_argmax_partial = nil;
    id<MTLComputePipelineState> argmax_reduce_pairs = nil;
    id<MTLComputePipelineState> argmax = nil;
};

void release_pipeline(id<MTLComputePipelineState>& pipeline) {
    if (pipeline != nil) {
        [pipeline release];
        pipeline = nil;
    }
}

void release_pipelines(Pipelines& pipelines) {
    release_pipeline(pipelines.matmul);
    release_pipeline(pipelines.matvec);
    release_pipeline(pipelines.small_m_gemm);
    release_pipeline(pipelines.small_m_gemm_tiled);
    release_pipeline(pipelines.embedding);
    release_pipeline(pipelines.add);
    release_pipeline(pipelines.add_rms_norm);
    release_pipeline(pipelines.rms_norm);
    release_pipeline(pipelines.rope);
    release_pipeline(pipelines.write_kv_cache);
    release_pipeline(pipelines.attention);
    release_pipeline(pipelines.swiglu);
    release_pipeline(pipelines.matvec_argmax_partial);
    release_pipeline(pipelines.argmax_reduce_pairs);
    release_pipeline(pipelines.argmax);
}

Status build_pipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    const char* name,
    id<MTLComputePipelineState>& out) {
    NSString* function_name = [NSString stringWithUTF8String:name];
    id<MTLFunction> function = [library newFunctionWithName:function_name];
    if (function == nil) {
        return Status::backend_error_status("Metal kernel function was not found");
    }

    NSError* error = nil;
    out = [device newComputePipelineStateWithFunction:function error:&error];
    [function release];
    if (out == nil) {
        return Status::backend_error_status("Metal compute pipeline creation failed");
    }
    return Status::success();
}

Status build_pipelines(id<MTLDevice> device, Pipelines& pipelines) {
    NSString* source = [NSString stringWithUTF8String:kMetalSource];
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    if (library == nil) {
        if (error != nil) {
            std::fprintf(
                stderr,
                "Metal library compilation failed: %s\n",
                [[error localizedDescription] UTF8String]);
        }
        return Status::backend_error_status("Metal library compilation failed");
    }

    Status status = build_pipeline(device, library, "matmul_f32", pipelines.matmul);
    if (status) {
        status = build_pipeline(device, library, "matvec_f32", pipelines.matvec);
    }
    if (status) {
        status = build_pipeline(device, library, "small_m_gemm_f32", pipelines.small_m_gemm);
    }
    if (status) {
        status = build_pipeline(
            device,
            library,
            "small_m_gemm_tiled_f32",
            pipelines.small_m_gemm_tiled);
    }
    if (status) {
        status = build_pipeline(device, library, "embedding_f32", pipelines.embedding);
    }
    if (status) {
        status = build_pipeline(device, library, "add_inplace_f32", pipelines.add);
    }
    if (status) {
        status = build_pipeline(
            device,
            library,
            "add_rms_norm_f32",
            pipelines.add_rms_norm);
    }
    if (status) {
        status = build_pipeline(device, library, "rms_norm_f32", pipelines.rms_norm);
    }
    if (status) {
        status = build_pipeline(device, library, "rope_f32", pipelines.rope);
    }
    if (status) {
        status = build_pipeline(device, library, "write_kv_cache_f32", pipelines.write_kv_cache);
    }
    if (status) {
        status = build_pipeline(device, library, "attention_f32", pipelines.attention);
    }
    if (status) {
        status = build_pipeline(device, library, "swiglu_f32", pipelines.swiglu);
    }
    if (status) {
        status = build_pipeline(
            device,
            library,
            "matvec_argmax_partial_f32",
            pipelines.matvec_argmax_partial);
    }
    if (status) {
        status = build_pipeline(
            device,
            library,
            "argmax_reduce_pairs_f32",
            pipelines.argmax_reduce_pairs);
    }
    if (status) {
        status = build_pipeline(device, library, "argmax_f32", pipelines.argmax);
    }

    [library release];
    if (!status) {
        release_pipelines(pipelines);
    }
    return status;
}

uint32_t checked_u32(int64_t value, const char* message) {
    if (value < 0 || value > UINT32_MAX) {
        panic(message, __FILE__, __LINE__);
    }
    return static_cast<uint32_t>(value);
}

NSUInteger dispatch_width(id<MTLComputePipelineState> pipeline, NSUInteger total) {
    return std::min<NSUInteger>(total, [pipeline maxTotalThreadsPerThreadgroup]);
}

bool metal_profile_enabled() {
    const char* value = std::getenv("TINYINFER_METAL_PROFILE");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

class MetalBackend final : public Backend {
public:
    MetalBackend(id<MTLDevice> device, id<MTLCommandQueue> queue, Pipelines pipelines)
        : device_(device),
          queue_(queue),
          pipelines_(pipelines),
          profile_enabled_(metal_profile_enabled()) {}

    ~MetalBackend() override {
        (void)flush();
        release_pipelines(pipelines_);
        if (argmax_buffer_ != nil) {
            [argmax_buffer_ release];
        }
        if (argmax_partial_buffer_ != nil) {
            [argmax_partial_buffer_ release];
        }
        [queue_ release];
        [device_ release];
    }

    Device device() const override {
        return {DeviceType::metal, 0};
    }

    Status alloc_arena(
        MemoryArena& arena,
        size_t bytes,
        MemoryKind kind) override {
        if (bytes == 0) {
            return Status::invalid_argument_status("Metal arena size must be non-zero");
        }
        if (bytes > static_cast<size_t>([device_ maxBufferLength])) {
            return Status::invalid_argument_status("Metal arena size exceeds maxBufferLength");
        }

        id<MTLBuffer> buffer =
            [device_ newBufferWithLength:static_cast<NSUInteger>(bytes)
                                  options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            return Status::backend_error_status("Metal arena allocation failed");
        }

        bind_arena(arena, (void*)buffer, bytes, kind);
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

        status = flush();
        if (!status) {
            return status;
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

        status = flush();
        if (!status) {
            return status;
        }
        std::memcpy(dst, data(src), bytes);
        return Status::success();
    }

    void matmul_out(
        const TensorView& out,
        const TensorView& x,
        const TensorView& w) override {
        MatmulParams params;
        params.m = checked_u32(x.dim(0), "Metal matmul m exceeds uint32");
        params.k = checked_u32(x.dim(1), "Metal matmul k exceeds uint32");
        params.n = checked_u32(w.dim(0), "Metal matmul n exceeds uint32");

        id<MTLComputeCommandEncoder> encoder = pending_encoder();
        id<MTLComputePipelineState> pipeline = pipelines_.matmul;
        if (params.m == 1) {
            pipeline = pipelines_.matvec;
        } else if (params.m <= 16) {
            pipeline = params.k % 128 == 0
                ? pipelines_.small_m_gemm_tiled
                : pipelines_.small_m_gemm;
        }
        if (pipeline == pipelines_.matvec) {
            profile_signpost(encoder, @"matvec_f32");
        } else if (pipeline == pipelines_.small_m_gemm_tiled) {
            profile_signpost(encoder, @"small_m_gemm_tiled_f32");
        } else if (pipeline == pipelines_.small_m_gemm) {
            profile_signpost(encoder, @"small_m_gemm_f32");
        } else {
            profile_signpost(encoder, @"matmul_f32");
        }
        [encoder setComputePipelineState:pipeline];
        set_tensor(encoder, out, 0);
        set_tensor(encoder, x, 1);
        set_tensor(encoder, w, 2);
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        if (params.m == 1) {
            constexpr NSUInteger simd_width = 32;
            constexpr NSUInteger outputs_per_threadgroup = 4;
            constexpr NSUInteger threads_per_threadgroup =
                outputs_per_threadgroup * simd_width;
            const NSUInteger threadgroups_for_n =
                (static_cast<NSUInteger>(params.n) + outputs_per_threadgroup - 1) /
                outputs_per_threadgroup;
            [encoder dispatchThreadgroups:MTLSizeMake(threadgroups_for_n, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threads_per_threadgroup, 1, 1)];
        } else if (params.m <= 16) {
            constexpr NSUInteger simd_width = 32;
            constexpr NSUInteger output_rows_per_tile = 4;
            constexpr NSUInteger output_cols_per_tile = 4;
            constexpr NSUInteger threads_per_threadgroup =
                output_cols_per_tile * simd_width;
            const NSUInteger threadgroups_for_n =
                (static_cast<NSUInteger>(params.n) + output_cols_per_tile - 1) /
                output_cols_per_tile;
            const NSUInteger threadgroups_for_m =
                (static_cast<NSUInteger>(params.m) + output_rows_per_tile - 1) /
                output_rows_per_tile;
            [encoder dispatchThreadgroups:
                    MTLSizeMake(threadgroups_for_n, threadgroups_for_m, 1)
                threadsPerThreadgroup:MTLSizeMake(threads_per_threadgroup, 1, 1)];
        } else {
            dispatch_1d(
                encoder,
                pipeline,
                static_cast<NSUInteger>(params.m) * params.n);
        }
    }

    void matmul_argmax(
        uint32_t& out_index,
        const TensorView& x,
        const TensorView& w) override {
        constexpr NSUInteger outputs_per_threadgroup = 4;
        constexpr NSUInteger matvec_threads_per_threadgroup = 128;
        constexpr NSUInteger reduce_threads_per_threadgroup = 256;

        MatmulParams matmul_params;
        matmul_params.m = checked_u32(x.dim(0), "Metal matmul_argmax m exceeds uint32");
        matmul_params.k = checked_u32(x.dim(1), "Metal matmul_argmax k exceeds uint32");
        matmul_params.n = checked_u32(w.dim(0), "Metal matmul_argmax n exceeds uint32");
        const NSUInteger partial_count =
            (static_cast<NSUInteger>(matmul_params.n) +
             outputs_per_threadgroup - 1) /
            outputs_per_threadgroup;

        ensure_argmax_buffer();
        ensure_argmax_partial_buffer(partial_count);

        id<MTLComputeCommandEncoder> encoder = pending_encoder();
        profile_signpost(encoder, @"matvec_argmax_partial_f32");
        [encoder setComputePipelineState:pipelines_.matvec_argmax_partial];
        [encoder setBuffer:argmax_partial_buffer_ offset:0 atIndex:0];
        set_tensor(encoder, x, 1);
        set_tensor(encoder, w, 2);
        [encoder setBytes:&matmul_params length:sizeof(matmul_params) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(partial_count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(matvec_threads_per_threadgroup, 1, 1)];

        ArgmaxParams argmax_params;
        argmax_params.count = checked_u32(
            static_cast<int64_t>(partial_count),
            "Metal argmax partial count exceeds uint32");
        profile_signpost(encoder, @"argmax_reduce_pairs_f32");
        [encoder setComputePipelineState:pipelines_.argmax_reduce_pairs];
        [encoder setBuffer:argmax_buffer_ offset:0 atIndex:0];
        [encoder setBuffer:argmax_partial_buffer_ offset:0 atIndex:1];
        [encoder setBytes:&argmax_params length:sizeof(argmax_params) atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(reduce_threads_per_threadgroup, 1, 1)];

        Status status = flush();
        if (!status) {
            panic("Metal fused matmul argmax failed", __FILE__, __LINE__);
        }
        out_index = *static_cast<uint32_t*>([argmax_buffer_ contents]);
    }

    void embedding_out(
        const TensorView& out,
        const TensorView& table,
        std::span<const uint32_t> token_ids) override {
        if (token_ids.empty()) {
            return;
        }

        id<MTLBuffer> token_buffer =
            [device_ newBufferWithBytes:token_ids.data()
                                 length:token_ids.size_bytes()
                                options:MTLResourceStorageModeShared];
        if (token_buffer == nil) {
            panic("Metal token buffer allocation failed", __FILE__, __LINE__);
        }

        EmbeddingParams params;
        params.token_count = checked_u32(
            static_cast<int64_t>(token_ids.size()),
            "Metal embedding token count exceeds uint32");
        params.hidden = checked_u32(table.dim(1), "Metal embedding hidden exceeds uint32");

        id<MTLComputeCommandEncoder> encoder = pending_encoder();
        profile_signpost(encoder, @"embedding_f32");
        [encoder setComputePipelineState:pipelines_.embedding];
        set_tensor(encoder, out, 0);
        set_tensor(encoder, table, 1);
        [encoder setBuffer:token_buffer offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        dispatch_1d(
            encoder,
            pipelines_.embedding,
            static_cast<NSUInteger>(params.token_count) * params.hidden);
        transient_buffers_.push_back(token_buffer);
    }

    void add_inplace(
        const TensorView& dst,
        const TensorView& src) override {
        AddParams params;
        params.count = checked_u32(dst.numel(), "Metal add count exceeds uint32");

        id<MTLComputeCommandEncoder> encoder = pending_encoder();
        profile_signpost(encoder, @"add_f32");
        [encoder setComputePipelineState:pipelines_.add];
        set_tensor(encoder, dst, 0);
        set_tensor(encoder, src, 1);
        [encoder setBytes:&params length:sizeof(params) atIndex:2];
        dispatch_1d(encoder, pipelines_.add, params.count);
    }

    void add_rms_norm_out(
        const TensorView& norm_out,
        const TensorView& residual_out,
        const TensorView& residual,
        const TensorView& weight,
        float eps) override {
        const NSUInteger row_bytes =
            static_cast<NSUInteger>(residual_out.dim(1)) * sizeof(float);
        if (row_bytes > [device_ maxThreadgroupMemoryLength]) {
            add_inplace(residual_out, residual);
            rms_norm_out(norm_out, residual_out, weight, eps);
            return;
        }

        RmsNormParams params;
        params.rows = checked_u32(
            residual_out.dim(0),
            "Metal fused add RMSNorm rows exceeds uint32");
        params.hidden = checked_u32(
            residual_out.dim(1),
            "Metal fused add RMSNorm hidden exceeds uint32");
        params.eps = eps;

        id<MTLComputeCommandEncoder> encoder = pending_encoder();
        profile_signpost(encoder, @"add_rms_norm_f32");
        [encoder setComputePipelineState:pipelines_.add_rms_norm];
        set_tensor(encoder, norm_out, 0);
        set_tensor(encoder, residual_out, 1);
        set_tensor(encoder, residual, 2);
        set_tensor(encoder, weight, 3);
        [encoder setBytes:&params length:sizeof(params) atIndex:4];
        [encoder setThreadgroupMemoryLength:row_bytes atIndex:0];
        constexpr NSUInteger threads_per_threadgroup = 256;
        [encoder dispatchThreadgroups:MTLSizeMake(params.rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads_per_threadgroup, 1, 1)];
    }

    void rms_norm_out(
        const TensorView& out,
        const TensorView& x,
        const TensorView& weight,
        float eps) override {
        RmsNormParams params;
        params.rows = checked_u32(x.dim(0), "Metal RMSNorm rows exceeds uint32");
        params.hidden = checked_u32(x.dim(1), "Metal RMSNorm hidden exceeds uint32");
        params.eps = eps;

        id<MTLComputeCommandEncoder> encoder = pending_encoder();
        profile_signpost(encoder, @"rms_norm_f32");
        [encoder setComputePipelineState:pipelines_.rms_norm];
        set_tensor(encoder, out, 0);
        set_tensor(encoder, x, 1);
        set_tensor(encoder, weight, 2);
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        constexpr NSUInteger threads_per_threadgroup = 256;
        [encoder dispatchThreadgroups:MTLSizeMake(params.rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads_per_threadgroup, 1, 1)];
    }

    void rope_inplace(
        const TensorView& q,
        const TensorView& k,
        uint32_t start_pos,
        float theta) override {
        RopeParams q_params;
        q_params.seq_len = checked_u32(q.dim(0), "Metal RoPE q seq_len exceeds uint32");
        q_params.heads = checked_u32(q.dim(1), "Metal RoPE q heads exceeds uint32");
        q_params.head_dim = checked_u32(q.dim(2), "Metal RoPE q head_dim exceeds uint32");
        q_params.start_pos = start_pos;
        q_params.theta = theta;

        RopeParams k_params;
        k_params.seq_len = checked_u32(k.dim(0), "Metal RoPE k seq_len exceeds uint32");
        k_params.heads = checked_u32(k.dim(1), "Metal RoPE k heads exceeds uint32");
        k_params.head_dim = checked_u32(k.dim(2), "Metal RoPE k head_dim exceeds uint32");
        k_params.start_pos = start_pos;
        k_params.theta = theta;

        id<MTLComputeCommandEncoder> encoder = pending_encoder();
        if (profile_enabled_) {
            [pending_command_buffer_ setLabel:
                start_pos == 0 ? @"tinyinfer.prefill" : @"tinyinfer.decode"];
        }
        profile_signpost(encoder, @"rope_q_f32");
        [encoder setComputePipelineState:pipelines_.rope];
        set_tensor(encoder, q, 0);
        [encoder setBytes:&q_params length:sizeof(q_params) atIndex:1];
        dispatch_1d(
            encoder,
            pipelines_.rope,
            static_cast<NSUInteger>(q_params.seq_len) *
                q_params.heads *
                (q_params.head_dim / 2));

        profile_signpost(encoder, @"rope_k_f32");
        set_tensor(encoder, k, 0);
        [encoder setBytes:&k_params length:sizeof(k_params) atIndex:1];
        dispatch_1d(
            encoder,
            pipelines_.rope,
            static_cast<NSUInteger>(k_params.seq_len) *
                k_params.heads *
                (k_params.head_dim / 2));
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
        WriteKVParams write_params;
        write_params.seq_len = checked_u32(k.dim(0), "Metal KV write seq_len exceeds uint32");
        write_params.n_kv_heads = checked_u32(k.dim(1), "Metal KV write heads exceeds uint32");
        write_params.max_seq_len = checked_u32(k_cache.dim(1), "Metal KV cache length exceeds uint32");
        write_params.head_dim = checked_u32(k.dim(2), "Metal KV head_dim exceeds uint32");
        write_params.start_pos = start_pos;

        AttentionParams attention_params;
        attention_params.seq_len = checked_u32(q.dim(0), "Metal attention seq_len exceeds uint32");
        attention_params.n_heads = checked_u32(q.dim(1), "Metal attention heads exceeds uint32");
        attention_params.n_kv_heads = checked_u32(k.dim(1), "Metal attention kv heads exceeds uint32");
        attention_params.max_seq_len = checked_u32(k_cache.dim(1), "Metal attention max_seq_len exceeds uint32");
        attention_params.head_dim = checked_u32(q.dim(2), "Metal attention head_dim exceeds uint32");
        attention_params.start_pos = start_pos;
        attention_params.kv_len = kv_len;

        id<MTLComputeCommandEncoder> encoder = pending_encoder();
        profile_signpost(encoder, @"write_kv_cache_f32");
        [encoder setComputePipelineState:pipelines_.write_kv_cache];
        set_tensor(encoder, k_cache, 0);
        set_tensor(encoder, v_cache, 1);
        set_tensor(encoder, k, 2);
        set_tensor(encoder, v, 3);
        [encoder setBytes:&write_params length:sizeof(write_params) atIndex:4];
        dispatch_1d(
            encoder,
            pipelines_.write_kv_cache,
            static_cast<NSUInteger>(write_params.seq_len) *
                write_params.n_kv_heads *
                write_params.head_dim);

        profile_signpost(encoder, @"attention_f32");
        [encoder setComputePipelineState:pipelines_.attention];
        set_tensor(encoder, out, 0);
        set_tensor(encoder, q, 1);
        set_tensor(encoder, k_cache, 2);
        set_tensor(encoder, v_cache, 3);
        [encoder setBytes:&attention_params length:sizeof(attention_params) atIndex:4];
        dispatch_1d(
            encoder,
            pipelines_.attention,
            static_cast<NSUInteger>(attention_params.seq_len) *
                attention_params.n_heads *
                attention_params.head_dim);
    }

    void swiglu_out(
        const TensorView& out,
        const TensorView& gate,
        const TensorView& up) override {
        UnaryParams params;
        params.count = checked_u32(out.numel(), "Metal SwiGLU count exceeds uint32");

        id<MTLComputeCommandEncoder> encoder = pending_encoder();
        profile_signpost(encoder, @"swiglu_f32");
        [encoder setComputePipelineState:pipelines_.swiglu];
        set_tensor(encoder, out, 0);
        set_tensor(encoder, gate, 1);
        set_tensor(encoder, up, 2);
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        dispatch_1d(encoder, pipelines_.swiglu, params.count);
    }

    void argmax(
        uint32_t& out_token,
        const TensorView& logits) override {
        ensure_argmax_buffer();
        *static_cast<uint32_t*>([argmax_buffer_ contents]) = 0;

        ArgmaxParams params;
        params.count = checked_u32(logits.numel(), "Metal argmax count exceeds uint32");

        id<MTLComputeCommandEncoder> encoder = pending_encoder();
        profile_signpost(encoder, @"argmax_f32");
        [encoder setComputePipelineState:pipelines_.argmax];
        [encoder setBuffer:argmax_buffer_ offset:0 atIndex:0];
        set_tensor(encoder, logits, 1);
        [encoder setBytes:&params length:sizeof(params) atIndex:2];
        dispatch_1d(encoder, pipelines_.argmax, 1);
        Status status = flush();
        if (!status) {
            panic("Metal command buffer execution failed", __FILE__, __LINE__);
        }

        out_token = *static_cast<uint32_t*>([argmax_buffer_ contents]);
    }

    Status synchronize() override {
        return flush();
    }

protected:
    void release_arena(MemoryArena& arena) noexcept override {
        (void)flush();
        id<MTLBuffer> buffer = metal_buffer(arena);
        if (buffer != nil) {
            [buffer release];
        }
    }

private:
    Status validate_copy_view(const TensorView& view, size_t bytes) const {
        Status status = validate_metal_contiguous(view);
        if (!status) {
            return status;
        }
        if (bytes > view.logical_nbytes()) {
            return Status::invalid_argument_status("copy exceeds tensor logical byte size");
        }
        return Status::success();
    }

    Status validate_metal_contiguous(const TensorView& view) const {
        if (!view.defined()) {
            return Status::invalid_argument_status("tensor view is not defined");
        }
        if (!owns_arena(*view.arena)) {
            return Status::invalid_argument_status("tensor view belongs to a different backend");
        }
        if (view.device().type != DeviceType::metal) {
            return Status::invalid_argument_status("tensor view is not on Metal");
        }
        if (!view.contiguous()) {
            return Status::invalid_argument_status("tensor view must be contiguous");
        }
        if (metal_buffer(*view.arena) == nil) {
            return Status::backend_error_status("Metal arena buffer is null");
        }
        return Status::success();
    }

    id<MTLComputeCommandEncoder> pending_encoder() {
        if (pending_encoder_ != nil) {
            return pending_encoder_;
        }

        pending_command_buffer_ = [[queue_ commandBuffer] retain];
        if (pending_command_buffer_ == nil) {
            panic("Metal command buffer creation failed", __FILE__, __LINE__);
        }
        if (profile_enabled_) {
            [pending_command_buffer_ setLabel:@"tinyinfer.inference"];
        }
        pending_encoder_ = [[pending_command_buffer_ computeCommandEncoder] retain];
        if (pending_encoder_ == nil) {
            [pending_command_buffer_ release];
            pending_command_buffer_ = nil;
            panic("Metal compute encoder creation failed", __FILE__, __LINE__);
        }
        if (profile_enabled_) {
            [pending_encoder_ setLabel:@"tinyinfer.compute"];
        }
        return pending_encoder_;
    }

    Status flush() {
        if (pending_command_buffer_ == nil) {
            return Status::success();
        }

        [pending_encoder_ endEncoding];
        [pending_encoder_ release];
        pending_encoder_ = nil;

        [pending_command_buffer_ commit];
        [pending_command_buffer_ waitUntilCompleted];
        const bool failed =
            [pending_command_buffer_ status] == MTLCommandBufferStatusError;
        if (profile_enabled_) {
            const CFTimeInterval gpu_start = [pending_command_buffer_ GPUStartTime];
            const CFTimeInterval gpu_end = [pending_command_buffer_ GPUEndTime];
            NSString* label = [pending_command_buffer_ label];
            if (gpu_start > 0.0 && gpu_end >= gpu_start) {
                std::fprintf(
                    stderr,
                    "metal_profile,label=%s,gpu_ms=%.3f\n",
                    label == nil ? "unlabeled" : [label UTF8String],
                    (gpu_end - gpu_start) * 1000.0);
            }
        }

        for (id<MTLBuffer> buffer : transient_buffers_) {
            [buffer release];
        }
        transient_buffers_.clear();

        [pending_command_buffer_ release];
        pending_command_buffer_ = nil;
        if (failed) {
            return Status::backend_error_status("Metal command buffer execution failed");
        }
        return Status::success();
    }

    void profile_signpost(
        id<MTLComputeCommandEncoder> encoder,
        NSString* label) const {
        if (profile_enabled_) {
            [encoder insertDebugSignpost:label];
        }
    }

    void dispatch_1d(
        id<MTLComputeCommandEncoder> encoder,
        id<MTLComputePipelineState> pipeline,
        NSUInteger total) {
        if (total == 0) {
            return;
        }
        const NSUInteger width = dispatch_width(pipeline, total);
        [encoder dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    }

    void set_tensor(
        id<MTLComputeCommandEncoder> encoder,
        const TensorView& view,
        NSUInteger index) {
        [encoder setBuffer:metal_buffer(*view.arena)
                    offset:static_cast<NSUInteger>(view.byte_offset)
                   atIndex:index];
    }

    void ensure_argmax_buffer() {
        if (argmax_buffer_ != nil) {
            return;
        }
        argmax_buffer_ = [device_ newBufferWithLength:sizeof(uint32_t)
                                             options:MTLResourceStorageModeShared];
        if (argmax_buffer_ == nil) {
            panic("Metal argmax buffer allocation failed", __FILE__, __LINE__);
        }
    }

    void ensure_argmax_partial_buffer(NSUInteger count) {
        const NSUInteger bytes = count * sizeof(ArgmaxPair);
        if (argmax_partial_buffer_ != nil &&
            [argmax_partial_buffer_ length] >= bytes) {
            return;
        }
        if (argmax_partial_buffer_ != nil) {
            [argmax_partial_buffer_ release];
        }
        argmax_partial_buffer_ =
            [device_ newBufferWithLength:bytes
                                 options:MTLResourceStorageModePrivate];
        if (argmax_partial_buffer_ == nil) {
            panic("Metal argmax partial buffer allocation failed", __FILE__, __LINE__);
        }
    }

    id<MTLBuffer> metal_buffer(const MemoryArena& arena) const {
        return (id<MTLBuffer>)arena_handle(arena);
    }

    uint8_t* data(const TensorView& view) {
        return static_cast<uint8_t*>([metal_buffer(*view.arena) contents]) +
            view.byte_offset;
    }

    const uint8_t* data(const TensorView& view) const {
        return static_cast<const uint8_t*>([metal_buffer(*view.arena) contents]) +
            view.byte_offset;
    }

    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    Pipelines pipelines_;
    id<MTLBuffer> argmax_buffer_ = nil;
    id<MTLBuffer> argmax_partial_buffer_ = nil;
    id<MTLCommandBuffer> pending_command_buffer_ = nil;
    id<MTLComputeCommandEncoder> pending_encoder_ = nil;
    std::vector<id<MTLBuffer>> transient_buffers_;
    bool profile_enabled_ = false;
};

#endif

}  // namespace

Result<std::unique_ptr<Backend>> create_metal_backend() {
#if TINYINFER_HAS_METAL
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        return {
            Status::backend_error_status("Metal device is not available"),
            nullptr,
        };
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
        [device release];
        return {
            Status::backend_error_status("Metal command queue creation failed"),
            nullptr,
        };
    }

    Pipelines pipelines;
    Status status = build_pipelines(device, pipelines);
    if (!status) {
        [queue release];
        [device release];
        return {status, nullptr};
    }

    return {
        Status::success(),
        std::make_unique<MetalBackend>(device, queue, pipelines),
    };
#else
    return {
        Status::unimplemented_status("Metal framework is not available"),
        nullptr,
    };
#endif
}

}  // namespace tinyinfer
