#include "tinyinfer/llama.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace tinyinfer;

namespace {

int g_failures = 0;

template <class T>
concept HasNativeHandle = requires(const T& value) {
    value.native_handle();
};

static_assert(!HasNativeHandle<TensorView>);
static_assert(!HasNativeHandle<MemoryArena>);

#define EXPECT_TRUE(expr)                                                        \
    do {                                                                         \
        if (!(expr)) {                                                           \
            std::cerr << __FILE__ << ":" << __LINE__ << " failed: " #expr "\n"; \
            ++g_failures;                                                        \
        }                                                                        \
    } while (false)

#define EXPECT_EQ(lhs, rhs)                                                      \
    do {                                                                         \
        const auto lhs_value = (lhs);                                             \
        const auto rhs_value = (rhs);                                             \
        if (!(lhs_value == rhs_value)) {                                          \
            std::cerr << __FILE__ << ":" << __LINE__ << " failed: " #lhs         \
                      << " == " #rhs << " (" << lhs_value << " vs "             \
                      << rhs_value << ")\n";                                     \
            ++g_failures;                                                        \
        }                                                                        \
    } while (false)

#define EXPECT_NEAR(lhs, rhs, tolerance)                                         \
    do {                                                                         \
        const double lhs_value = static_cast<double>(lhs);                        \
        const double rhs_value = static_cast<double>(rhs);                        \
        const double tolerance_value = static_cast<double>(tolerance);            \
        if (std::fabs(lhs_value - rhs_value) > tolerance_value) {                 \
            std::cerr << __FILE__ << ":" << __LINE__ << " failed: " #lhs         \
                      << " ~= " #rhs << " (" << lhs_value << " vs "             \
                      << rhs_value << ")\n";                                     \
            ++g_failures;                                                        \
        }                                                                        \
    } while (false)

class FakeBackend final : public Backend {
public:
    std::vector<std::string> calls;

    Device device() const override {
        return {DeviceType::cpu, 0};
    }

    Status alloc_arena(MemoryArena& arena, size_t bytes, MemoryKind kind) override {
        calls.push_back("alloc_arena");
        bind_arena(arena, reinterpret_cast<void*>(0x1000), bytes, kind);
        return Status::success();
    }

    Status copy_from_host(const TensorView& dst, const void* src, size_t bytes) override {
        (void)dst;
        (void)src;
        (void)bytes;
        calls.push_back("copy_from_host");
        return Status::success();
    }

    Status copy_to_host(void* dst, const TensorView& src, size_t bytes) override {
        (void)dst;
        (void)src;
        (void)bytes;
        calls.push_back("copy_to_host");
        return Status::success();
    }

    Status matmul_out(const TensorView& out, const TensorView& x, const TensorView& w) override {
        (void)out;
        (void)x;
        (void)w;
        calls.push_back("matmul");
        return Status::success();
    }

    Status embedding_out(
        const TensorView& out,
        const TensorView& table,
        std::span<const uint32_t> token_ids) override {
        (void)out;
        (void)table;
        (void)token_ids;
        calls.push_back("embedding");
        return Status::success();
    }

    Status add_inplace(const TensorView& dst, const TensorView& src) override {
        (void)dst;
        (void)src;
        calls.push_back("add");
        return Status::success();
    }

    Status rms_norm_out(
        const TensorView& out,
        const TensorView& x,
        const TensorView& weight,
        float eps) override {
        (void)out;
        (void)x;
        (void)weight;
        (void)eps;
        calls.push_back("rms_norm");
        return Status::success();
    }

    Status rope_inplace(
        const TensorView& q,
        const TensorView& k,
        uint32_t start_pos,
        float theta) override {
        (void)q;
        (void)k;
        (void)start_pos;
        (void)theta;
        calls.push_back("rope");
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
        (void)out;
        (void)q;
        (void)k;
        (void)v;
        (void)k_cache;
        (void)v_cache;
        (void)start_pos;
        (void)kv_len;
        calls.push_back("attention");
        return Status::success();
    }

    Status swiglu_out(
        const TensorView& out,
        const TensorView& gate,
        const TensorView& up) override {
        (void)out;
        (void)gate;
        (void)up;
        calls.push_back("swiglu");
        return Status::success();
    }

    Status argmax(uint32_t& out_token, const TensorView& logits) override {
        (void)logits;
        calls.push_back("argmax");
        out_token = 42;
        return Status::success();
    }

    Status synchronize() override {
        calls.push_back("synchronize");
        return Status::success();
    }

protected:
    Status release_arena(MemoryArena& arena) override {
        (void)arena;
        calls.push_back("release_arena");
        return Status::success();
    }
};

void copy_f32_to_tensor(Backend& backend, const TensorView& tensor, const float* values) {
    EXPECT_TRUE(backend.copy_from_host(tensor, values, tensor.logical_nbytes()));
}

void expect_f32_tensor_near(
    Backend& backend,
    const TensorView& tensor,
    const std::vector<float>& expected,
    float tolerance = 1e-5f) {
    std::vector<float> actual(expected.size(), 0.0f);
    EXPECT_EQ(tensor.logical_nbytes(), expected.size() * sizeof(float));
    EXPECT_TRUE(backend.copy_to_host(actual.data(), tensor, tensor.logical_nbytes()));
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], tolerance);
    }
}

void test_config_validation() {
    LlamaConfig config = LlamaConfig::tinyllama_1_1b();
    EXPECT_TRUE(config.validate());
    EXPECT_EQ(config.head_dim(), 64u);
    EXPECT_EQ(config.kv_group_size(), 8u);

    config.hidden_size = 2050;
    EXPECT_EQ(config.validate().code, Status::invalid_config);
}

void test_shape_stride_tensor_view_helpers() {
    FakeBackend backend;
    MemoryArena arena;
    EXPECT_TRUE(backend.alloc_arena(arena, 4096, MemoryKind::workspace));

    Shape shape = make_shape({2, 3, 4});
    EXPECT_TRUE(shape.valid());
    EXPECT_EQ(shape.ndim, 3u);
    EXPECT_EQ(shape.dim(1), 3);
    EXPECT_EQ(shape.numel(), 24);

    Strides strides = contiguous_strides(shape);
    EXPECT_EQ(strides.stride(0), 12);
    EXPECT_EQ(strides.stride(1), 4);
    EXPECT_EQ(strides.stride(2), 1);

    Result<TensorView> tensor = arena.alloc(shape, DType::f32);
    EXPECT_TRUE(tensor.status);
    EXPECT_TRUE(tensor.value.defined());
    EXPECT_EQ(tensor.value.logical_nbytes(), 96u);
    EXPECT_EQ(tensor.value.storage_span_nbytes(), 96u);
    EXPECT_TRUE(tensor.value.contiguous());
    EXPECT_EQ(arena.used(), 96u);

    Result<TensorView> second = arena.alloc(make_shape({2, 3}), DType::f16);
    EXPECT_TRUE(second.status);
    EXPECT_EQ(second.value.byte_offset, 128u);
    EXPECT_EQ(second.value.logical_nbytes(), 12u);
    EXPECT_EQ(arena.used(), 140u);
}

void test_tensor_view_edge_cases() {
    Shape scalar = make_shape({});
    EXPECT_TRUE(scalar.valid());
    EXPECT_EQ(scalar.ndim, 0u);
    EXPECT_EQ(scalar.numel(), 1);

    TensorView scalar_view;
    scalar_view.dtype = DType::f16;
    scalar_view.shape = scalar;
    scalar_view.strides = contiguous_strides(scalar);
    EXPECT_EQ(scalar_view.logical_nbytes(), 2u);
    EXPECT_EQ(scalar_view.storage_span_nbytes(), 2u);
    EXPECT_TRUE(scalar_view.contiguous());
    EXPECT_TRUE(!scalar_view.defined());

    Shape zero_dim = make_shape({2, 0});
    EXPECT_TRUE(!zero_dim.valid());
    EXPECT_EQ(zero_dim.numel(), 0);
    EXPECT_EQ(contiguous_strides(zero_dim).ndim, 0u);

    Shape too_many_dims = make_shape({1, 2, 3, 4, 5, 6, 7, 8, 9});
    EXPECT_TRUE(!too_many_dims.valid());
    EXPECT_EQ(too_many_dims.dim(8), 0);

    Strides too_many_strides;
    too_many_strides.ndim = 9;
    EXPECT_EQ(too_many_strides.stride(8), 0);

    Shape huge;
    huge.ndim = 2;
    huge.dims[0] = std::numeric_limits<int64_t>::max();
    huge.dims[1] = 2;
    EXPECT_TRUE(!huge.valid());
    EXPECT_EQ(huge.numel(), 0);

    TensorView gapped;
    gapped.dtype = DType::f16;
    gapped.shape = make_shape({2, 3});
    gapped.strides.ndim = 2;
    gapped.strides.values[0] = 4;
    gapped.strides.values[1] = 1;
    EXPECT_EQ(gapped.logical_nbytes(), 12u);
    EXPECT_EQ(gapped.storage_span_nbytes(), 14u);
    EXPECT_TRUE(!gapped.contiguous());

    gapped.strides.values[0] = -1;
    EXPECT_EQ(gapped.storage_span_nbytes(), 0u);
}

void test_arena_reset_reuses_memory() {
    FakeBackend backend;
    MemoryArena arena;
    EXPECT_TRUE(backend.alloc_arena(arena, 1024, MemoryKind::workspace));

    Result<TensorView> a = arena.alloc(make_shape({4, 4}), DType::f32);
    EXPECT_TRUE(a.status);
    EXPECT_EQ(a.value.byte_offset, 0u);
    EXPECT_EQ(arena.used(), 64u);

    arena.reset();
    Result<TensorView> b = arena.alloc(make_shape({4, 4}), DType::f32);
    EXPECT_TRUE(b.status);
    EXPECT_EQ(b.value.byte_offset, 0u);
    EXPECT_EQ(arena.used(), 64u);
}

void test_cpu_backend_memory() {
    Result<std::unique_ptr<Backend>> backend = create_cpu_backend();
    EXPECT_TRUE(backend.status);
    EXPECT_TRUE(backend.value != nullptr);
    EXPECT_TRUE(backend.value->device().type == DeviceType::cpu);

    MemoryArena zero;
    Status zero_status = backend.value->alloc_arena(zero, 0, MemoryKind::workspace);
    EXPECT_EQ(zero_status.code, Status::invalid_argument);
    EXPECT_TRUE(!zero.defined());

    MemoryArena arena;
    EXPECT_TRUE(backend.value->alloc_arena(arena, 256, MemoryKind::host));
    EXPECT_TRUE(arena.defined());
    EXPECT_EQ(arena.capacity(), 256u);
    EXPECT_TRUE(arena.device().type == DeviceType::cpu);

    MemoryArena moved = std::move(arena);
    EXPECT_TRUE(!arena.defined());
    EXPECT_TRUE(moved.defined());

    Result<TensorView> tensor = moved.alloc(make_shape({4}), DType::f32);
    EXPECT_TRUE(tensor.status);
    EXPECT_EQ(tensor.value.byte_offset % 64, 0u);

    const float input[] = {1.0f, -2.0f, 3.5f, 4.0f};
    float output[] = {0.0f, 0.0f, 0.0f, 0.0f};

    EXPECT_TRUE(backend.value->copy_from_host(tensor.value, input, sizeof(input)));
    EXPECT_TRUE(backend.value->copy_to_host(output, tensor.value, sizeof(output)));
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(output[i], input[i]);
    }

    EXPECT_TRUE(backend.value->copy_from_host(tensor.value, nullptr, 0));
    EXPECT_EQ(
        backend.value->copy_from_host(tensor.value, input, sizeof(input) + 1).code,
        Status::invalid_argument);

    TensorView non_contiguous = tensor.value;
    non_contiguous.strides.values[0] = 2;
    EXPECT_EQ(
        backend.value->copy_from_host(non_contiguous, input, sizeof(input)).code,
        Status::invalid_argument);

    FakeBackend foreign_backend;
    MemoryArena foreign_arena;
    EXPECT_TRUE(foreign_backend.alloc_arena(foreign_arena, 256, MemoryKind::host));
    Result<TensorView> foreign_tensor = foreign_arena.alloc(make_shape({4}), DType::f32);
    EXPECT_TRUE(foreign_tensor.status);
    EXPECT_EQ(
        backend.value->copy_from_host(foreign_tensor.value, input, sizeof(input)).code,
        Status::invalid_argument);

    EXPECT_TRUE(backend.value->synchronize());
}

void test_cpu_backend_argmax() {
    Result<std::unique_ptr<Backend>> backend = create_cpu_backend();
    EXPECT_TRUE(backend.status);

    MemoryArena arena;
    EXPECT_TRUE(backend.value->alloc_arena(arena, 2048, MemoryKind::workspace));

    Result<TensorView> logits = arena.alloc(make_shape({5}), DType::f32);
    EXPECT_TRUE(logits.status);
    const float input[] = {1.0f, 3.0f, 3.0f, -1.0f, 2.0f};
    EXPECT_TRUE(backend.value->copy_from_host(logits.value, input, sizeof(input)));

    uint32_t token = 99;
    EXPECT_TRUE(backend.value->argmax(token, logits.value));
    EXPECT_EQ(token, 1u);

    Result<TensorView> batched_logits = arena.alloc(make_shape({1, 4}), DType::f32);
    EXPECT_TRUE(batched_logits.status);
    const float batched_input[] = {-4.0f, -2.0f, 0.5f, 0.25f};
    EXPECT_TRUE(backend.value->copy_from_host(
        batched_logits.value,
        batched_input,
        sizeof(batched_input)));
    EXPECT_TRUE(backend.value->argmax(token, batched_logits.value));
    EXPECT_EQ(token, 2u);

    Result<TensorView> f16_logits = arena.alloc(make_shape({4}), DType::f16);
    EXPECT_TRUE(f16_logits.status);
    EXPECT_EQ(backend.value->argmax(token, f16_logits.value).code, Status::unimplemented);

    TensorView non_contiguous = logits.value;
    non_contiguous.strides.values[0] = 2;
    EXPECT_EQ(backend.value->argmax(token, non_contiguous).code, Status::invalid_argument);

    Result<TensorView> bad_rank = arena.alloc(make_shape({1, 1, 5}), DType::f32);
    EXPECT_TRUE(bad_rank.status);
    EXPECT_EQ(backend.value->argmax(token, bad_rank.value).code, Status::invalid_argument);

    FakeBackend foreign_backend;
    MemoryArena foreign_arena;
    EXPECT_TRUE(foreign_backend.alloc_arena(foreign_arena, 256, MemoryKind::workspace));
    Result<TensorView> foreign_logits = foreign_arena.alloc(make_shape({4}), DType::f32);
    EXPECT_TRUE(foreign_logits.status);
    EXPECT_EQ(backend.value->argmax(token, foreign_logits.value).code, Status::invalid_argument);
}

void test_cpu_backend_matmul() {
    Result<std::unique_ptr<Backend>> backend = create_cpu_backend();
    EXPECT_TRUE(backend.status);

    MemoryArena arena;
    EXPECT_TRUE(backend.value->alloc_arena(arena, 4096, MemoryKind::workspace));

    Result<TensorView> x = arena.alloc(make_shape({2, 3}), DType::f32);
    Result<TensorView> w = arena.alloc(make_shape({2, 3}), DType::f32);
    Result<TensorView> out = arena.alloc(make_shape({2, 2}), DType::f32);
    EXPECT_TRUE(x.status);
    EXPECT_TRUE(w.status);
    EXPECT_TRUE(out.status);

    const float x_values[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    };
    const float w_values[] = {
        10.0f, 20.0f, 30.0f,
        1.0f, 0.0f, -1.0f,
    };
    copy_f32_to_tensor(*backend.value, x.value, x_values);
    copy_f32_to_tensor(*backend.value, w.value, w_values);

    EXPECT_TRUE(backend.value->matmul_out(out.value, x.value, w.value));
    expect_f32_tensor_near(*backend.value, out.value, {140.0f, -2.0f, 320.0f, -2.0f});

    Result<TensorView> out3 = arena.alloc(make_shape({2, 1, 2}), DType::f32);
    EXPECT_TRUE(out3.status);
    EXPECT_TRUE(backend.value->matmul_out(out3.value, x.value, w.value));
    expect_f32_tensor_near(*backend.value, out3.value, {140.0f, -2.0f, 320.0f, -2.0f});

    Result<TensorView> bad_w = arena.alloc(make_shape({3, 2}), DType::f32);
    Result<TensorView> bad_out = arena.alloc(make_shape({2, 3}), DType::f32);
    EXPECT_TRUE(bad_w.status);
    EXPECT_TRUE(bad_out.status);
    EXPECT_EQ(backend.value->matmul_out(out.value, x.value, bad_w.value).code, Status::invalid_argument);
    EXPECT_EQ(backend.value->matmul_out(bad_out.value, x.value, w.value).code, Status::invalid_argument);
}

void test_cpu_backend_embedding_and_add() {
    Result<std::unique_ptr<Backend>> backend = create_cpu_backend();
    EXPECT_TRUE(backend.status);

    MemoryArena arena;
    EXPECT_TRUE(backend.value->alloc_arena(arena, 4096, MemoryKind::workspace));

    Result<TensorView> table = arena.alloc(make_shape({4, 3}), DType::f32);
    Result<TensorView> out = arena.alloc(make_shape({2, 3}), DType::f32);
    EXPECT_TRUE(table.status);
    EXPECT_TRUE(out.status);

    const float table_values[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f,
    };
    copy_f32_to_tensor(*backend.value, table.value, table_values);

    const uint32_t tokens[] = {2, 0};
    EXPECT_TRUE(backend.value->embedding_out(out.value, table.value, tokens));
    expect_f32_tensor_near(*backend.value, out.value, {
        7.0f, 8.0f, 9.0f,
        1.0f, 2.0f, 3.0f,
    });

    const uint32_t bad_tokens[] = {4};
    EXPECT_EQ(
        backend.value->embedding_out(out.value, table.value, bad_tokens).code,
        Status::invalid_argument);

    Result<TensorView> add_src = arena.alloc(make_shape({2, 3}), DType::f32);
    EXPECT_TRUE(add_src.status);
    const float add_values[] = {
        0.5f, -1.0f, 2.0f,
        3.0f, 4.0f, -5.0f,
    };
    copy_f32_to_tensor(*backend.value, add_src.value, add_values);

    EXPECT_TRUE(backend.value->add_inplace(out.value, add_src.value));
    expect_f32_tensor_near(*backend.value, out.value, {
        7.5f, 7.0f, 11.0f,
        4.0f, 6.0f, -2.0f,
    });

    Result<TensorView> bad_add = arena.alloc(make_shape({3}), DType::f32);
    EXPECT_TRUE(bad_add.status);
    EXPECT_EQ(
        backend.value->add_inplace(out.value, bad_add.value).code,
        Status::invalid_argument);
}

void test_cpu_backend_rms_norm_and_swiglu() {
    Result<std::unique_ptr<Backend>> backend = create_cpu_backend();
    EXPECT_TRUE(backend.status);

    MemoryArena arena;
    EXPECT_TRUE(backend.value->alloc_arena(arena, 4096, MemoryKind::workspace));

    Result<TensorView> x = arena.alloc(make_shape({2, 3}), DType::f32);
    Result<TensorView> weight = arena.alloc(make_shape({3}), DType::f32);
    Result<TensorView> out = arena.alloc(make_shape({2, 3}), DType::f32);
    EXPECT_TRUE(x.status);
    EXPECT_TRUE(weight.status);
    EXPECT_TRUE(out.status);

    const float x_values[] = {
        1.0f, 2.0f, 2.0f,
        3.0f, 0.0f, 4.0f,
    };
    const float weight_values[] = {1.0f, 0.5f, -1.0f};
    copy_f32_to_tensor(*backend.value, x.value, x_values);
    copy_f32_to_tensor(*backend.value, weight.value, weight_values);

    constexpr float eps = 1e-5f;
    EXPECT_TRUE(backend.value->rms_norm_out(out.value, x.value, weight.value, eps));

    std::vector<float> expected_rms;
    for (int row = 0; row < 2; ++row) {
        const float* x_row = x_values + row * 3;
        float mean_sq = 0.0f;
        for (int i = 0; i < 3; ++i) {
            mean_sq += x_row[i] * x_row[i];
        }
        mean_sq /= 3.0f;
        const float scale = 1.0f / std::sqrt(mean_sq + eps);
        for (int i = 0; i < 3; ++i) {
            expected_rms.push_back(x_row[i] * scale * weight_values[i]);
        }
    }
    expect_f32_tensor_near(*backend.value, out.value, expected_rms);
    EXPECT_EQ(
        backend.value->rms_norm_out(out.value, x.value, weight.value, 0.0f).code,
        Status::invalid_argument);

    Result<TensorView> gate = arena.alloc(make_shape({3}), DType::f32);
    Result<TensorView> up = arena.alloc(make_shape({3}), DType::f32);
    Result<TensorView> swiglu = arena.alloc(make_shape({3}), DType::f32);
    EXPECT_TRUE(gate.status);
    EXPECT_TRUE(up.status);
    EXPECT_TRUE(swiglu.status);

    const float gate_values[] = {0.0f, 1.0f, -1.0f};
    const float up_values[] = {2.0f, 3.0f, 4.0f};
    copy_f32_to_tensor(*backend.value, gate.value, gate_values);
    copy_f32_to_tensor(*backend.value, up.value, up_values);

    EXPECT_TRUE(backend.value->swiglu_out(swiglu.value, gate.value, up.value));
    expect_f32_tensor_near(
        *backend.value,
        swiglu.value,
        {
            0.0f,
            (1.0f / (1.0f + std::exp(-1.0f))) * 3.0f,
            (-1.0f / (1.0f + std::exp(1.0f))) * 4.0f,
        });
}

void test_cpu_backend_rope() {
    Result<std::unique_ptr<Backend>> backend = create_cpu_backend();
    EXPECT_TRUE(backend.status);

    MemoryArena arena;
    EXPECT_TRUE(backend.value->alloc_arena(arena, 4096, MemoryKind::workspace));

    Result<TensorView> q = arena.alloc(make_shape({1, 1, 4}), DType::f32);
    Result<TensorView> k = arena.alloc(make_shape({1, 1, 4}), DType::f32);
    EXPECT_TRUE(q.status);
    EXPECT_TRUE(k.status);

    const float q_values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float k_values[] = {-1.0f, 0.5f, 2.0f, -0.25f};
    copy_f32_to_tensor(*backend.value, q.value, q_values);
    copy_f32_to_tensor(*backend.value, k.value, k_values);

    constexpr float theta = 10000.0f;
    EXPECT_TRUE(backend.value->rope_inplace(q.value, k.value, 1, theta));

    auto expected_rope = [](const float* values) {
        std::vector<float> expected(values, values + 4);
        for (int i = 0; i < 2; ++i) {
            const float exponent = static_cast<float>(2 * i) / 4.0f;
            const float angle = 1.0f / std::pow(10000.0f, exponent);
            const float c = std::cos(angle);
            const float s = std::sin(angle);
            const float x0 = values[i];
            const float x1 = values[i + 2];
            expected[i] = x0 * c - x1 * s;
            expected[i + 2] = x1 * c + x0 * s;
        }
        return expected;
    };

    expect_f32_tensor_near(*backend.value, q.value, expected_rope(q_values));
    expect_f32_tensor_near(*backend.value, k.value, expected_rope(k_values));
    EXPECT_EQ(backend.value->rope_inplace(q.value, k.value, 0, 0.0f).code, Status::invalid_argument);
}

void test_cpu_backend_attention() {
    Result<std::unique_ptr<Backend>> backend = create_cpu_backend();
    EXPECT_TRUE(backend.status);

    MemoryArena arena;
    EXPECT_TRUE(backend.value->alloc_arena(arena, 8192, MemoryKind::workspace));

    Result<TensorView> q = arena.alloc(make_shape({2, 2, 2}), DType::f32);
    Result<TensorView> k = arena.alloc(make_shape({2, 1, 2}), DType::f32);
    Result<TensorView> v = arena.alloc(make_shape({2, 1, 2}), DType::f32);
    Result<TensorView> k_cache = arena.alloc(make_shape({1, 4, 2}), DType::f32);
    Result<TensorView> v_cache = arena.alloc(make_shape({1, 4, 2}), DType::f32);
    Result<TensorView> out = arena.alloc(make_shape({2, 4}), DType::f32);
    EXPECT_TRUE(q.status);
    EXPECT_TRUE(k.status);
    EXPECT_TRUE(v.status);
    EXPECT_TRUE(k_cache.status);
    EXPECT_TRUE(v_cache.status);
    EXPECT_TRUE(out.status);

    const float q_values[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
    };
    const float k_values[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };
    const float v_values[] = {
        10.0f, 100.0f,
        20.0f, 200.0f,
    };
    const float zero_cache[] = {
        0.0f, 0.0f,
        0.0f, 0.0f,
        0.0f, 0.0f,
        0.0f, 0.0f,
    };
    copy_f32_to_tensor(*backend.value, q.value, q_values);
    copy_f32_to_tensor(*backend.value, k.value, k_values);
    copy_f32_to_tensor(*backend.value, v.value, v_values);
    copy_f32_to_tensor(*backend.value, k_cache.value, zero_cache);
    copy_f32_to_tensor(*backend.value, v_cache.value, zero_cache);

    EXPECT_TRUE(backend.value->attention_out(
        out.value,
        q.value,
        k.value,
        v.value,
        k_cache.value,
        v_cache.value,
        0,
        2));

    const float a = 1.0f / std::sqrt(2.0f);
    const float exp_a = std::exp(a);
    const float head0_w0 = exp_a / (exp_a + 1.0f);
    const float head0_w1 = 1.0f / (exp_a + 1.0f);
    const float head1_w0 = 1.0f / (1.0f + exp_a);
    const float head1_w1 = exp_a / (1.0f + exp_a);
    expect_f32_tensor_near(
        *backend.value,
        out.value,
        {
            10.0f, 100.0f,
            10.0f, 100.0f,
            head0_w0 * 10.0f + head0_w1 * 20.0f,
            head0_w0 * 100.0f + head0_w1 * 200.0f,
            head1_w0 * 10.0f + head1_w1 * 20.0f,
            head1_w0 * 100.0f + head1_w1 * 200.0f,
        },
        1e-4f);
    expect_f32_tensor_near(
        *backend.value,
        k_cache.value,
        {
            1.0f, 0.0f,
            0.0f, 1.0f,
            0.0f, 0.0f,
            0.0f, 0.0f,
        });
    expect_f32_tensor_near(
        *backend.value,
        v_cache.value,
        {
            10.0f, 100.0f,
            20.0f, 200.0f,
            0.0f, 0.0f,
            0.0f, 0.0f,
        });
    EXPECT_EQ(
        backend.value->attention_out(
            out.value,
            q.value,
            k.value,
            v.value,
            k_cache.value,
            v_cache.value,
            0,
            1)
            .code,
        Status::invalid_argument);
}

void test_engine_flow_with_fake_backend() {
    FakeBackend backend;
    LlamaConfig config = LlamaConfig::demo();

    Result<LlamaInferEngine> engine = LlamaInferEngine::create(backend, config, 16);
    EXPECT_TRUE(engine.status);
    EXPECT_EQ(engine.value.max_seq_len(), 16u);
    EXPECT_EQ(engine.value.seq_len(), 0u);

    const TokenId prompt[] = {1, 2, 3};
    TokenId next = 0;
    EXPECT_TRUE(engine.value.prefill(prompt, next));
    EXPECT_EQ(engine.value.seq_len(), 3u);
    EXPECT_EQ(next, 42u);

    EXPECT_TRUE(engine.value.decode_one(4, next));
    EXPECT_EQ(engine.value.seq_len(), 4u);
    EXPECT_EQ(next, 42u);

    TokenId output[5] = {};
    uint32_t output_count = 0;
    EXPECT_TRUE(engine.value.generate(prompt, output, GenerateConfig{2, 2, true}, output_count));
    EXPECT_EQ(output_count, 5u);
    EXPECT_EQ(output[0], 1u);
    EXPECT_EQ(output[1], 2u);
    EXPECT_EQ(output[2], 3u);
    EXPECT_EQ(output[3], 42u);
    EXPECT_EQ(output[4], 42u);

    bool saw_attention = false;
    bool saw_argmax = false;
    bool saw_embedding = false;
    bool saw_add = false;
    uint32_t arena_allocs = 0;
    for (const std::string& call : backend.calls) {
        saw_attention = saw_attention || call == "attention";
        saw_argmax = saw_argmax || call == "argmax";
        saw_embedding = saw_embedding || call == "embedding";
        saw_add = saw_add || call == "add";
        arena_allocs += call == "alloc_arena" ? 1u : 0u;
    }
    EXPECT_EQ(arena_allocs, 3u);
    EXPECT_TRUE(saw_embedding);
    EXPECT_TRUE(saw_attention);
    EXPECT_TRUE(saw_add);
    EXPECT_TRUE(saw_argmax);
}

}  // namespace

int main() {
    test_config_validation();
    test_shape_stride_tensor_view_helpers();
    test_tensor_view_edge_cases();
    test_arena_reset_reuses_memory();
    test_cpu_backend_memory();
    test_cpu_backend_argmax();
    test_cpu_backend_matmul();
    test_cpu_backend_embedding_and_add();
    test_cpu_backend_rms_norm_and_swiglu();
    test_cpu_backend_rope();
    test_cpu_backend_attention();
    test_engine_flow_with_fake_backend();

    if (g_failures != 0) {
        std::cerr << g_failures << " test failure(s)\n";
        return EXIT_FAILURE;
    }

    std::cout << "tinyinfer tests passed\n";
    return EXIT_SUCCESS;
}
