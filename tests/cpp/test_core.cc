#include "tinyinfer/model_loader.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
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
    std::vector<std::vector<float>> copied_f32;

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
        calls.push_back("copy_from_host");
        if (src != nullptr && bytes != 0 && bytes % sizeof(float) == 0) {
            const float* values = static_cast<const float*>(src);
            copied_f32.emplace_back(values, values + bytes / sizeof(float));
        }
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

std::filesystem::path temp_test_path(const char* name) {
    return std::filesystem::temp_directory_path() / name;
}

void write_text_file(const std::filesystem::path& path, const std::string& text) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    EXPECT_TRUE(static_cast<bool>(file));
    file.write(text.data(), static_cast<std::streamsize>(text.size()));
    EXPECT_TRUE(static_cast<bool>(file));
}

std::string config_json_for(const LlamaConfig& config) {
    return std::string("{\n") +
        "  \"model_type\": \"llama\",\n" +
        "  \"hidden_act\": \"silu\",\n" +
        "  \"attention_bias\": false,\n" +
        "  \"tie_word_embeddings\": false,\n" +
        "  \"pretraining_tp\": 1,\n" +
        "  \"rope_scaling\": null,\n" +
        "  \"num_hidden_layers\": " + std::to_string(config.n_layers) + ",\n" +
        "  \"hidden_size\": " + std::to_string(config.hidden_size) + ",\n" +
        "  \"intermediate_size\": " + std::to_string(config.intermediate_size) + ",\n" +
        "  \"num_attention_heads\": " + std::to_string(config.n_heads) + ",\n" +
        "  \"num_key_value_heads\": " + std::to_string(config.n_kv_heads) + ",\n" +
        "  \"vocab_size\": " + std::to_string(config.vocab_size) + ",\n" +
        "  \"max_position_embeddings\": " + std::to_string(config.max_seq_len) + ",\n" +
        "  \"rms_norm_eps\": 0.00001,\n" +
        "  \"rope_theta\": 10000.0\n" +
        "}\n";
}

uint16_t f32_to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(float));
    return static_cast<uint16_t>(bits >> 16);
}

void append_u64_le(std::vector<unsigned char>& out, uint64_t value) {
    for (uint32_t i = 0; i < 8; ++i) {
        out.push_back(static_cast<unsigned char>((value >> (8 * i)) & 0xffU));
    }
}

uint64_t fixture_numel(const std::vector<int64_t>& shape) {
    uint64_t total = 1;
    for (int64_t dim : shape) {
        total *= static_cast<uint64_t>(dim);
    }
    return total;
}

std::string shape_json(const std::vector<int64_t>& shape) {
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            out += ",";
        }
        out += std::to_string(shape[i]);
    }
    out += "]";
    return out;
}

struct FixtureTensor {
    std::string key;
    std::vector<int64_t> shape;
    std::string dtype = "F32";
    float base = 0.0f;
};

enum class FixtureMode {
    valid,
    missing_lm_head,
    wrong_embedding_shape,
    unsupported_embedding_dtype,
};

void add_fixture_tensor(
    std::vector<FixtureTensor>& tensors,
    const std::string& key,
    std::vector<int64_t> shape,
    float base) {
    tensors.push_back({key, std::move(shape), "F32", base});
}

std::vector<FixtureTensor> build_fixture_tensors(
    const LlamaConfig& config,
    FixtureMode mode) {
    std::vector<FixtureTensor> tensors;
    tensors.reserve(3 + static_cast<size_t>(config.n_layers) * 9);

    const int64_t hidden = config.hidden_size;
    const int64_t inter = config.intermediate_size;
    const int64_t kv_dim = config.n_kv_heads * config.head_dim();

    FixtureTensor embedding;
    embedding.key = "model.embed_tokens.weight";
    embedding.shape = {
        mode == FixtureMode::wrong_embedding_shape ? 1 : static_cast<int64_t>(config.vocab_size),
        hidden,
    };
    embedding.dtype = mode == FixtureMode::unsupported_embedding_dtype ? "I64" : "BF16";
    embedding.base = 1.0f;
    tensors.push_back(std::move(embedding));

    add_fixture_tensor(tensors, "model.norm.weight", {hidden}, 100.0f);
    if (mode != FixtureMode::missing_lm_head) {
        add_fixture_tensor(
            tensors,
            "lm_head.weight",
            {static_cast<int64_t>(config.vocab_size), hidden},
            200.0f);
    }

    for (uint32_t i = 0; i < config.n_layers; ++i) {
        const std::string prefix = "model.layers." + std::to_string(i);
        const float base = 1000.0f + static_cast<float>(i) * 100.0f;
        add_fixture_tensor(tensors, prefix + ".input_layernorm.weight", {hidden}, base + 1.0f);
        add_fixture_tensor(tensors, prefix + ".self_attn.q_proj.weight", {hidden, hidden}, base + 2.0f);
        add_fixture_tensor(tensors, prefix + ".self_attn.k_proj.weight", {kv_dim, hidden}, base + 3.0f);
        add_fixture_tensor(tensors, prefix + ".self_attn.v_proj.weight", {kv_dim, hidden}, base + 4.0f);
        add_fixture_tensor(tensors, prefix + ".self_attn.o_proj.weight", {hidden, hidden}, base + 5.0f);
        add_fixture_tensor(tensors, prefix + ".post_attention_layernorm.weight", {hidden}, base + 6.0f);
        add_fixture_tensor(tensors, prefix + ".mlp.gate_proj.weight", {inter, hidden}, base + 7.0f);
        add_fixture_tensor(tensors, prefix + ".mlp.up_proj.weight", {inter, hidden}, base + 8.0f);
        add_fixture_tensor(tensors, prefix + ".mlp.down_proj.weight", {hidden, inter}, base + 9.0f);
    }

    return tensors;
}

std::vector<unsigned char> fixture_tensor_bytes(const FixtureTensor& tensor) {
    const uint64_t count = fixture_numel(tensor.shape);
    std::vector<unsigned char> bytes;
    if (tensor.dtype == "F32") {
        bytes.resize(static_cast<size_t>(count) * sizeof(float));
        for (uint64_t i = 0; i < count; ++i) {
            const float value = tensor.base + static_cast<float>(i);
            std::memcpy(
                bytes.data() + static_cast<size_t>(i) * sizeof(float),
                &value,
                sizeof(float));
        }
    } else if (tensor.dtype == "BF16") {
        bytes.resize(static_cast<size_t>(count) * sizeof(uint16_t));
        for (uint64_t i = 0; i < count; ++i) {
            const float value = tensor.base + static_cast<float>(i);
            const uint16_t bf16 = f32_to_bf16(value);
            std::memcpy(
                bytes.data() + static_cast<size_t>(i) * sizeof(uint16_t),
                &bf16,
                sizeof(uint16_t));
        }
    } else {
        bytes.resize(static_cast<size_t>(count) * sizeof(uint64_t), 0);
    }
    return bytes;
}

void write_safetensors_fixture(
    const std::filesystem::path& path,
    const LlamaConfig& config,
    FixtureMode mode) {
    std::vector<FixtureTensor> tensors = build_fixture_tensors(config, mode);

    std::string header = "{";
    std::vector<unsigned char> data;
    uint64_t offset = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        const FixtureTensor& tensor = tensors[i];
        std::vector<unsigned char> bytes = fixture_tensor_bytes(tensor);
        const uint64_t begin = offset;
        const uint64_t end = begin + bytes.size();

        if (i != 0) {
            header += ",";
        }
        header += "\n  \"" + tensor.key + "\": {";
        header += "\"dtype\": \"" + tensor.dtype + "\", ";
        header += "\"shape\": " + shape_json(tensor.shape) + ", ";
        header += "\"data_offsets\": [" + std::to_string(begin) + "," + std::to_string(end) + "]";
        header += "}";

        data.insert(data.end(), bytes.begin(), bytes.end());
        offset = end;
    }
    header += "\n}";

    std::vector<unsigned char> prefix;
    append_u64_le(prefix, static_cast<uint64_t>(header.size()));

    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    EXPECT_TRUE(static_cast<bool>(file));
    file.write(reinterpret_cast<const char*>(prefix.data()), static_cast<std::streamsize>(prefix.size()));
    file.write(header.data(), static_cast<std::streamsize>(header.size()));
    file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    EXPECT_TRUE(static_cast<bool>(file));
}

void test_config_validation() {
    LlamaConfig config = LlamaConfig::tinyllama_1_1b();
    EXPECT_TRUE(config.validate());
    EXPECT_EQ(config.head_dim(), 64u);
    EXPECT_EQ(config.kv_group_size(), 8u);

    config.hidden_size = 2050;
    EXPECT_EQ(config.validate().code, Status::invalid_config);
}

void test_model_loader_config_json() {
    const LlamaConfig expected = LlamaConfig::demo();
    const std::filesystem::path path = temp_test_path("tinyinfer_config_valid.json");
    write_text_file(path, config_json_for(expected));

    Result<LlamaConfig> loaded = load_llama_config_json(path.c_str());
    EXPECT_TRUE(loaded.status);
    EXPECT_EQ(loaded.value.n_layers, expected.n_layers);
    EXPECT_EQ(loaded.value.hidden_size, expected.hidden_size);
    EXPECT_EQ(loaded.value.intermediate_size, expected.intermediate_size);
    EXPECT_EQ(loaded.value.n_heads, expected.n_heads);
    EXPECT_EQ(loaded.value.n_kv_heads, expected.n_kv_heads);
    EXPECT_EQ(loaded.value.vocab_size, expected.vocab_size);
    EXPECT_EQ(loaded.value.max_seq_len, expected.max_seq_len);
    EXPECT_NEAR(loaded.value.rms_eps, expected.rms_eps, 1e-9f);
    EXPECT_NEAR(loaded.value.rope_theta, expected.rope_theta, 1e-5f);

    const std::filesystem::path bad_bias_path = temp_test_path("tinyinfer_config_bad_bias.json");
    std::string bad_bias = config_json_for(expected);
    const std::string from = "\"attention_bias\": false";
    const std::string to = "\"attention_bias\": true";
    const size_t pos = bad_bias.find(from);
    EXPECT_TRUE(pos != std::string::npos);
    bad_bias.replace(pos, from.size(), to);
    write_text_file(bad_bias_path, bad_bias);
    EXPECT_EQ(load_llama_config_json(bad_bias_path.c_str()).status.code, Status::invalid_config);

    const std::filesystem::path missing_path = temp_test_path("tinyinfer_config_missing.json");
    write_text_file(missing_path, "{\"model_type\":\"llama\"}\n");
    EXPECT_EQ(load_llama_config_json(missing_path.c_str()).status.code, Status::invalid_config);
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

void test_model_loader_safetensors_with_fake_backend() {
    FakeBackend backend;
    LlamaConfig config;
    config.n_layers = 1;
    config.hidden_size = 4;
    config.intermediate_size = 8;
    config.n_heads = 2;
    config.n_kv_heads = 1;
    config.vocab_size = 8;
    config.max_seq_len = 8;
    config.rms_eps = 1e-5f;
    config.rope_theta = 10000.0f;
    EXPECT_TRUE(config.validate());

    Result<LlamaInferEngine> engine = LlamaInferEngine::create(backend, config, 8);
    EXPECT_TRUE(engine.status);

    const std::filesystem::path path = temp_test_path("tinyinfer_valid.safetensors");
    write_safetensors_fixture(path, config, FixtureMode::valid);

    Status status = load_llama_safetensors(engine.value, path.c_str());
    EXPECT_TRUE(status);

    const size_t expected_copies = 3 + static_cast<size_t>(config.n_layers) * 9;
    EXPECT_EQ(backend.copied_f32.size(), expected_copies);
    EXPECT_TRUE(!backend.copied_f32.empty());
    if (!backend.copied_f32.empty()) {
        EXPECT_EQ(backend.copied_f32[0].size(), static_cast<size_t>(config.vocab_size * config.hidden_size));
        EXPECT_NEAR(backend.copied_f32[0][0], 1.0f, 0.0f);
        EXPECT_NEAR(backend.copied_f32[0][1], 2.0f, 0.0f);
        EXPECT_NEAR(backend.copied_f32[0][2], 3.0f, 0.0f);
        EXPECT_NEAR(backend.copied_f32[0][3], 4.0f, 0.0f);
    }

    bool saw_synchronize = false;
    for (const std::string& call : backend.calls) {
        saw_synchronize = saw_synchronize || call == "synchronize";
    }
    EXPECT_TRUE(saw_synchronize);
}

void test_model_loader_rejects_bad_safetensors() {
    LlamaConfig config;
    config.n_layers = 1;
    config.hidden_size = 4;
    config.intermediate_size = 8;
    config.n_heads = 2;
    config.n_kv_heads = 1;
    config.vocab_size = 8;
    config.max_seq_len = 8;
    config.rms_eps = 1e-5f;
    config.rope_theta = 10000.0f;
    EXPECT_TRUE(config.validate());

    {
        FakeBackend backend;
        Result<LlamaInferEngine> engine = LlamaInferEngine::create(backend, config, 8);
        EXPECT_TRUE(engine.status);
        const std::filesystem::path path = temp_test_path("tinyinfer_missing.safetensors");
        write_safetensors_fixture(path, config, FixtureMode::missing_lm_head);
        EXPECT_EQ(load_llama_safetensors(engine.value, path.c_str()).code, Status::invalid_argument);
        EXPECT_TRUE(backend.copied_f32.empty());
    }

    {
        FakeBackend backend;
        Result<LlamaInferEngine> engine = LlamaInferEngine::create(backend, config, 8);
        EXPECT_TRUE(engine.status);
        const std::filesystem::path path = temp_test_path("tinyinfer_wrong_shape.safetensors");
        write_safetensors_fixture(path, config, FixtureMode::wrong_embedding_shape);
        EXPECT_EQ(load_llama_safetensors(engine.value, path.c_str()).code, Status::invalid_argument);
        EXPECT_TRUE(backend.copied_f32.empty());
    }

    {
        FakeBackend backend;
        Result<LlamaInferEngine> engine = LlamaInferEngine::create(backend, config, 8);
        EXPECT_TRUE(engine.status);
        const std::filesystem::path path = temp_test_path("tinyinfer_unsupported_dtype.safetensors");
        write_safetensors_fixture(path, config, FixtureMode::unsupported_embedding_dtype);
        EXPECT_EQ(load_llama_safetensors(engine.value, path.c_str()).code, Status::unimplemented);
        EXPECT_TRUE(backend.copied_f32.empty());
    }
}

}  // namespace

int main() {
    test_config_validation();
    test_model_loader_config_json();
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
    test_model_loader_safetensors_with_fake_backend();
    test_model_loader_rejects_bad_safetensors();

    if (g_failures != 0) {
        std::cerr << g_failures << " test failure(s)\n";
        return EXIT_FAILURE;
    }

    std::cout << "tinyinfer tests passed\n";
    return EXIT_SUCCESS;
}
