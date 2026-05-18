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

    Result<Stream> new_stream() override {
        calls.push_back("new_stream");
        return {Status::success(), make_stream()};
    }

    Result<Tensor> empty(const Shape& shape, DType dtype) override {
        calls.push_back("empty");
        return {Status::success(), make_tensor(shape, dtype, nullptr, shape.numel() * dtype_size(dtype))};
    }

    Status copy_from_host(Tensor& dst, const void* src, size_t bytes, Stream& stream) override {
        (void)dst;
        (void)src;
        (void)bytes;
        (void)stream;
        calls.push_back("copy_from_host");
        return Status::success();
    }

    Status copy_to_host(void* dst, const Tensor& src, size_t bytes, Stream& stream) override {
        (void)dst;
        (void)src;
        (void)bytes;
        (void)stream;
        calls.push_back("copy_to_host");
        return Status::success();
    }

    Status matmul_out(Tensor& out, const Tensor& x, const Tensor& w, Stream& stream) override {
        (void)x;
        (void)w;
        (void)stream;
        calls.push_back("matmul");
        out = make_tensor(make_shape({1, 1}), DType::f32, nullptr, 4);
        return Status::success();
    }

    Status rms_norm_out(
        Tensor& out,
        const Tensor& x,
        const Tensor& weight,
        float eps,
        Stream& stream) override {
        (void)x;
        (void)weight;
        (void)eps;
        (void)stream;
        calls.push_back("rms_norm");
        out = make_tensor(make_shape({1, 1}), DType::f32, nullptr, 4);
        return Status::success();
    }

    Status rope_inplace(Tensor& q, Tensor& k, uint32_t start_pos, Stream& stream) override {
        (void)q;
        (void)k;
        (void)start_pos;
        (void)stream;
        calls.push_back("rope");
        return Status::success();
    }

    Status attention_out(
        Tensor& out,
        const Tensor& q,
        const Tensor& k_cache,
        const Tensor& v_cache,
        uint32_t kv_len,
        Stream& stream) override {
        (void)q;
        (void)k_cache;
        (void)v_cache;
        (void)kv_len;
        (void)stream;
        calls.push_back("attention");
        out = make_tensor(make_shape({1, 1}), DType::f32, nullptr, 4);
        return Status::success();
    }

    Status swiglu_out(Tensor& out, const Tensor& gate, const Tensor& up, Stream& stream) override {
        (void)gate;
        (void)up;
        (void)stream;
        calls.push_back("swiglu");
        out = make_tensor(make_shape({1, 1}), DType::f32, nullptr, 4);
        return Status::success();
    }

    Status argmax(uint32_t& out_token, const Tensor& logits, Stream& stream) override {
        (void)logits;
        (void)stream;
        calls.push_back("argmax");
        out_token = 42;
        return Status::success();
    }

    Status synchronize(Stream& stream) override {
        (void)stream;
        calls.push_back("synchronize");
        return Status::success();
    }

protected:
    Status release_storage(Storage& storage) override {
        (void)storage;
        calls.push_back("release_storage");
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

void test_shape_stride_tensor_helpers() {
    Shape shape = make_shape({2, 3, 4});
    EXPECT_TRUE(shape.valid());
    EXPECT_EQ(shape.ndim, 3u);
    EXPECT_EQ(shape.dim(1), 3);
    EXPECT_EQ(shape.numel(), 24);

    Strides strides = contiguous_strides(shape);
    EXPECT_EQ(strides.stride(0), 12);
    EXPECT_EQ(strides.stride(1), 4);
    EXPECT_EQ(strides.stride(2), 1);

    Tensor tensor;
    tensor.dtype = DType::f32;
    tensor.shape = shape;
    tensor.strides = strides;
    EXPECT_EQ(tensor.nbytes(), 96u);
    EXPECT_TRUE(tensor.contiguous());
}

void test_parameter_enumeration() {
    LlamaModel model;
    model.config = LlamaConfig::demo();
    model.layers.resize(model.config.n_layers);

    std::vector<NamedTensor> params = model.parameters();
    EXPECT_EQ(params.size(), 21u);
    EXPECT_EQ(params[0].name, std::string("token_embedding"));
    EXPECT_EQ(params[3].name, std::string("layers.0.attn_norm"));
    EXPECT_EQ(params.back().name, std::string("layers.1.down_proj"));
}

void test_runner_flow_with_fake_backend() {
    FakeBackend backend;
    LlamaModel model;
    model.config = LlamaConfig::demo();
    model.layers.resize(model.config.n_layers);

    LlamaRunner runner(backend, model);

    KVCache cache;
    EXPECT_TRUE(runner.init_kv_cache(cache, 16));
    EXPECT_EQ(cache.max_seq_len, 16u);
    EXPECT_EQ(cache.seq_len, 0u);

    Tensor logits;
    const TokenId prompt[] = {1, 2, 3};
    EXPECT_TRUE(runner.prefill(prompt, cache, logits));
    EXPECT_EQ(cache.seq_len, 3u);

    TokenId next = 0;
    EXPECT_TRUE(runner.decode_one(4, cache, logits, next));
    EXPECT_EQ(cache.seq_len, 4u);
    EXPECT_EQ(next, 42u);

    bool saw_attention = false;
    bool saw_argmax = false;
    for (const std::string& call : backend.calls) {
        saw_attention = saw_attention || call == "attention";
        saw_argmax = saw_argmax || call == "argmax";
    }
    EXPECT_TRUE(saw_attention);
    EXPECT_TRUE(saw_argmax);
}

}  // namespace

int main() {
    test_config_validation();
    test_shape_stride_tensor_helpers();
    test_parameter_enumeration();
    test_runner_flow_with_fake_backend();

    if (g_failures != 0) {
        std::cerr << g_failures << " test failure(s)\n";
        return EXIT_FAILURE;
    }

    std::cout << "tinyinfer tests passed\n";
    return EXIT_SUCCESS;
}
