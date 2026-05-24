#include "tinyinfer/llama.h"

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

    Status rope_inplace(const TensorView& q, const TensorView& k, uint32_t start_pos) override {
        (void)q;
        (void)k;
        (void)start_pos;
        calls.push_back("rope");
        return Status::success();
    }

    Status attention_out(
        const TensorView& out,
        const TensorView& q,
        const TensorView& k_cache,
        const TensorView& v_cache,
        uint32_t kv_len) override {
        (void)out;
        (void)q;
        (void)k_cache;
        (void)v_cache;
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

    uint32_t token = 0;
    EXPECT_EQ(backend.value->argmax(token, tensor.value).code, Status::unimplemented);
    EXPECT_TRUE(backend.value->synchronize());
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
    uint32_t arena_allocs = 0;
    for (const std::string& call : backend.calls) {
        saw_attention = saw_attention || call == "attention";
        saw_argmax = saw_argmax || call == "argmax";
        arena_allocs += call == "alloc_arena" ? 1u : 0u;
    }
    EXPECT_EQ(arena_allocs, 3u);
    EXPECT_TRUE(saw_attention);
    EXPECT_TRUE(saw_argmax);
}

}  // namespace

int main() {
    test_config_validation();
    test_shape_stride_tensor_view_helpers();
    test_tensor_view_edge_cases();
    test_arena_reset_reuses_memory();
    test_cpu_backend_memory();
    test_engine_flow_with_fake_backend();

    if (g_failures != 0) {
        std::cerr << g_failures << " test failure(s)\n";
        return EXIT_FAILURE;
    }

    std::cout << "tinyinfer tests passed\n";
    return EXIT_SUCCESS;
}
