#include "tinyinfer/backend.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>

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
    uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) {
        return;
    }

    const uint base = row * p.hidden;
    float sum_sq = 0.0f;
    for (uint i = 0; i < p.hidden; ++i) {
        const float value = x[base + i];
        sum_sq += value * value;
    }

    const float mean_sq = sum_sq / float(p.hidden);
    const float scale = rsqrt(mean_sq + p.eps);
    for (uint i = 0; i < p.hidden; ++i) {
        out[base + i] = x[base + i] * scale * weight[i];
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

struct Pipelines {
    id<MTLComputePipelineState> matmul = nil;
    id<MTLComputePipelineState> embedding = nil;
    id<MTLComputePipelineState> add = nil;
    id<MTLComputePipelineState> rms_norm = nil;
    id<MTLComputePipelineState> rope = nil;
    id<MTLComputePipelineState> write_kv_cache = nil;
    id<MTLComputePipelineState> attention = nil;
    id<MTLComputePipelineState> swiglu = nil;
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
    release_pipeline(pipelines.embedding);
    release_pipeline(pipelines.add);
    release_pipeline(pipelines.rms_norm);
    release_pipeline(pipelines.rope);
    release_pipeline(pipelines.write_kv_cache);
    release_pipeline(pipelines.attention);
    release_pipeline(pipelines.swiglu);
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
        status = build_pipeline(device, library, "embedding_f32", pipelines.embedding);
    }
    if (status) {
        status = build_pipeline(device, library, "add_inplace_f32", pipelines.add);
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

class MetalBackend final : public Backend {
public:
    MetalBackend(id<MTLDevice> device, id<MTLCommandQueue> queue, Pipelines pipelines)
        : device_(device),
          queue_(queue),
          pipelines_(pipelines) {}

    ~MetalBackend() override {
        release_pipelines(pipelines_);
        if (argmax_buffer_ != nil) {
            [argmax_buffer_ release];
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
        MatmulParams params;
        params.m = checked_u32(x.dim(0), "Metal matmul m exceeds uint32");
        params.k = checked_u32(x.dim(1), "Metal matmul k exceeds uint32");
        params.n = checked_u32(w.dim(0), "Metal matmul n exceeds uint32");

        id<MTLCommandBuffer> command_buffer = begin_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipelines_.matmul];
        set_tensor(encoder, out, 0);
        set_tensor(encoder, x, 1);
        set_tensor(encoder, w, 2);
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        dispatch_1d(encoder, pipelines_.matmul, static_cast<NSUInteger>(params.m) * params.n);
        [encoder endEncoding];
        commit_and_wait(command_buffer);
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

        id<MTLCommandBuffer> command_buffer = begin_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipelines_.embedding];
        set_tensor(encoder, out, 0);
        set_tensor(encoder, table, 1);
        [encoder setBuffer:token_buffer offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        dispatch_1d(
            encoder,
            pipelines_.embedding,
            static_cast<NSUInteger>(params.token_count) * params.hidden);
        [encoder endEncoding];
        commit_and_wait(command_buffer);

        [token_buffer release];
    }

    void add_inplace(
        const TensorView& dst,
        const TensorView& src) override {
        AddParams params;
        params.count = checked_u32(dst.numel(), "Metal add count exceeds uint32");

        id<MTLCommandBuffer> command_buffer = begin_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipelines_.add];
        set_tensor(encoder, dst, 0);
        set_tensor(encoder, src, 1);
        [encoder setBytes:&params length:sizeof(params) atIndex:2];
        dispatch_1d(encoder, pipelines_.add, params.count);
        [encoder endEncoding];
        commit_and_wait(command_buffer);
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

        id<MTLCommandBuffer> command_buffer = begin_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipelines_.rms_norm];
        set_tensor(encoder, out, 0);
        set_tensor(encoder, x, 1);
        set_tensor(encoder, weight, 2);
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        dispatch_1d(encoder, pipelines_.rms_norm, params.rows);
        [encoder endEncoding];
        commit_and_wait(command_buffer);
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

        id<MTLCommandBuffer> command_buffer = begin_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipelines_.rope];
        set_tensor(encoder, q, 0);
        [encoder setBytes:&q_params length:sizeof(q_params) atIndex:1];
        dispatch_1d(
            encoder,
            pipelines_.rope,
            static_cast<NSUInteger>(q_params.seq_len) *
                q_params.heads *
                (q_params.head_dim / 2));

        set_tensor(encoder, k, 0);
        [encoder setBytes:&k_params length:sizeof(k_params) atIndex:1];
        dispatch_1d(
            encoder,
            pipelines_.rope,
            static_cast<NSUInteger>(k_params.seq_len) *
                k_params.heads *
                (k_params.head_dim / 2));
        [encoder endEncoding];
        commit_and_wait(command_buffer);
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

        id<MTLCommandBuffer> command_buffer = begin_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
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
        [encoder endEncoding];
        commit_and_wait(command_buffer);
    }

    void swiglu_out(
        const TensorView& out,
        const TensorView& gate,
        const TensorView& up) override {
        UnaryParams params;
        params.count = checked_u32(out.numel(), "Metal SwiGLU count exceeds uint32");

        id<MTLCommandBuffer> command_buffer = begin_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipelines_.swiglu];
        set_tensor(encoder, out, 0);
        set_tensor(encoder, gate, 1);
        set_tensor(encoder, up, 2);
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        dispatch_1d(encoder, pipelines_.swiglu, params.count);
        [encoder endEncoding];
        commit_and_wait(command_buffer);
    }

    void argmax(
        uint32_t& out_token,
        const TensorView& logits) override {
        ensure_argmax_buffer();
        *static_cast<uint32_t*>([argmax_buffer_ contents]) = 0;

        ArgmaxParams params;
        params.count = checked_u32(logits.numel(), "Metal argmax count exceeds uint32");

        id<MTLCommandBuffer> command_buffer = begin_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipelines_.argmax];
        [encoder setBuffer:argmax_buffer_ offset:0 atIndex:0];
        set_tensor(encoder, logits, 1);
        [encoder setBytes:&params length:sizeof(params) atIndex:2];
        dispatch_1d(encoder, pipelines_.argmax, 1);
        [encoder endEncoding];
        commit_and_wait(command_buffer);

        out_token = *static_cast<uint32_t*>([argmax_buffer_ contents]);
    }

    Status synchronize() override {
        return Status::success();
    }

protected:
    void release_arena(MemoryArena& arena) noexcept override {
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

    id<MTLCommandBuffer> begin_command_buffer() {
        id<MTLCommandBuffer> command_buffer = [queue_ commandBuffer];
        if (command_buffer == nil) {
            panic("Metal command buffer creation failed", __FILE__, __LINE__);
        }
        return command_buffer;
    }

    void commit_and_wait(id<MTLCommandBuffer> command_buffer) {
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer status] == MTLCommandBufferStatusError) {
            panic("Metal command buffer execution failed", __FILE__, __LINE__);
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
