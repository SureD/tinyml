#include "tinyinfer/llama.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace tinyinfer;

namespace {

int g_failures = 0;

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
    test_arena_reset_reuses_memory();
    test_engine_flow_with_fake_backend();

    if (g_failures != 0) {
        std::cerr << g_failures << " test failure(s)\n";
        return EXIT_FAILURE;
    }

    std::cout << "tinyinfer tests passed\n";
    return EXIT_SUCCESS;
}
