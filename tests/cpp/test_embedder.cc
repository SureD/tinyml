#include "tinyinfer/embedder.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace tinyinfer {
void dequantized_row_for_test(const float* weight, int64_t cols, int bits, float* out);
}

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

std::string fixture_path(const char* name) {
    return std::string(TINYINFER_EMBEDDER_FIXTURE_DIR) + "/" + name;
}

void write_text(const std::string& path, const std::string& text) {
    std::ofstream file(path, std::ios::binary);
    file << text;
}

bool load_tokens(const std::string& path, std::vector<std::vector<uint32_t>>& sequences) {
    std::ifstream file(path);
    if (!file) {
        return false;
    }
    size_t count = 0;
    file >> count;
    sequences.resize(count);
    for (size_t i = 0; i < count; ++i) {
        size_t length = 0;
        file >> length;
        sequences[i].resize(length);
        for (size_t t = 0; t < length; ++t) {
            file >> sequences[i][t];
        }
    }
    return static_cast<bool>(file);
}

bool load_f32(const std::string& path, std::vector<float>& values) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    file.seekg(0, std::ios::end);
    const std::streampos size = file.tellg();
    if (size < 0 || static_cast<size_t>(size) % sizeof(float) != 0) {
        return false;
    }
    file.seekg(0, std::ios::beg);
    values.resize(static_cast<size_t>(size) / sizeof(float));
    if (!values.empty()) {
        file.read(reinterpret_cast<char*>(values.data()), size);
    }
    return static_cast<bool>(file);
}

double cosine(const float* a, const float* b, size_t n) {
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    for (size_t i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    return dot / std::sqrt(na * nb);
}

void test_official_config() {
    const std::string path = fixture_path("embeddinggemma2/config.json");
    Result<EmbeddingGemma2Config> config = load_embeddinggemma2_config_json(path.c_str());
    EXPECT_TRUE(config.status);
    if (!config.status) {
        std::cerr << "official config: " << config.status.message << "\n";
        return;
    }

    const EmbeddingGemma2Config& text = config.value;
    EXPECT_TRUE(text.n_layers == 24);
    EXPECT_TRUE(text.hidden_size == 512);
    EXPECT_TRUE(text.intermediate_size == 2048);
    EXPECT_TRUE(text.n_heads == 4);
    EXPECT_TRUE(text.vocab_size == 262144);
    EXPECT_TRUE(text.embedding_dim == 768);
    EXPECT_TRUE(text.ple_dim == 512);
    EXPECT_TRUE(text.sliding_window == 512);
    EXPECT_TRUE(text.max_position_embeddings == 262144);
    EXPECT_TRUE(std::fabs(text.rms_eps - 1e-6f) < 1e-12f);
    EXPECT_TRUE(text.layers.size() == 24);

    for (uint32_t i = 0; i < 24; ++i) {
        const bool full = (i % 6) == 5;
        EXPECT_TRUE(text.layers[i].full_attention == full);
        if (full) {
            EXPECT_TRUE(text.layers[i].head_dim == 512);
            EXPECT_TRUE(text.layers[i].n_kv_heads == 1);
            EXPECT_TRUE(std::fabs(text.layers[i].rope_theta - 1000000.0f) < 1e-3f);
            EXPECT_TRUE(text.q_dim(i) == 2048);
            EXPECT_TRUE(text.kv_dim(i) == 512);
        } else {
            EXPECT_TRUE(text.layers[i].head_dim == 256);
            EXPECT_TRUE(text.layers[i].n_kv_heads == 2);
            EXPECT_TRUE(std::fabs(text.layers[i].rope_theta - 10000.0f) < 1e-3f);
            EXPECT_TRUE(text.q_dim(i) == 1024);
            EXPECT_TRUE(text.kv_dim(i) == 512);
        }
    }
}

void test_config_rejects() {
    const std::string llama = "/tmp/tinyinfer_embedder_llama.json";
    write_text(llama, R"({"model_type":"llama","hidden_act":"silu"})");
    Result<EmbeddingGemma2Config> llama_config = load_embeddinggemma2_config_json(llama.c_str());
    EXPECT_TRUE(!llama_config.status);
    EXPECT_TRUE(llama_config.status.code == Status::invalid_config);

    const std::string activation = "/tmp/tinyinfer_embedder_activation.json";
    write_text(
        activation,
        R"({
          "model_type":"embedding_gemma2_text",
          "hidden_activation":"silu",
          "num_hidden_layers":1,
          "hidden_size":8,
          "intermediate_size":16,
          "num_attention_heads":2,
          "num_key_value_heads":2,
          "head_dim":4,
          "vocab_size":8,
          "embedding_dim":4,
          "hidden_size_per_layer_input":4,
          "sliding_window":2,
          "max_position_embeddings":16,
          "rms_norm_eps":1e-6,
          "layer_types":["full_attention"],
          "rope_parameters":{
            "sliding_attention":{"rope_type":"default","rope_theta":10000},
            "full_attention":{"rope_type":"default","rope_theta":1000000}
          },
          "per_layer_config":{}
        })");
    Result<EmbeddingGemma2Config> bad_activation =
        load_embeddinggemma2_config_json(activation.c_str());
    EXPECT_TRUE(!bad_activation.status);

    const std::string missing = "/tmp/tinyinfer_embedder_missing_layers.json";
    write_text(missing, R"({"model_type":"embedding_gemma2","text_config":{"model_type":"embedding_gemma2_text"}})");
    Result<EmbeddingGemma2Config> missing_layers = load_embeddinggemma2_config_json(missing.c_str());
    EXPECT_TRUE(!missing_layers.status);

    const std::string fp16 = "/tmp/tinyinfer_embedder_fp16.json";
    write_text(
        fp16,
        R"({
          "model_type":"embedding_gemma2_text",
          "dtype":"float16",
          "hidden_activation":"gelu_pytorch_tanh",
          "num_hidden_layers":1,
          "hidden_size":8,
          "intermediate_size":16,
          "num_attention_heads":2,
          "num_key_value_heads":2,
          "head_dim":4,
          "vocab_size":8,
          "embedding_dim":4,
          "hidden_size_per_layer_input":4,
          "sliding_window":2,
          "max_position_embeddings":16,
          "rms_norm_eps":1e-6,
          "layer_types":["full_attention"],
          "rope_parameters":{
            "sliding_attention":{"rope_type":"default","rope_theta":10000},
            "full_attention":{"rope_type":"default","rope_theta":1000000}
          },
          "per_layer_config":{}
        })");
    Result<EmbeddingGemma2Config> fp16_config = load_embeddinggemma2_config_json(fp16.c_str());
    EXPECT_TRUE(!fp16_config.status);
}

void test_tiny_reference() {
    const std::string dir = fixture_path("embeddinggemma2_tiny");
    Result<std::unique_ptr<Backend>> backend = create_cpu_backend();
    EXPECT_TRUE(backend.status);
    if (!backend.status) {
        return;
    }

    Result<EmbeddingGemma2Engine> engine = load_embeddinggemma2_from_hf_dir(
        *backend.value,
        dir.c_str(),
        8);
    EXPECT_TRUE(engine.status);
    if (!engine.status) {
        std::cerr << "tiny load: " << engine.status.message << "\n";
        return;
    }
    EXPECT_TRUE(engine.value.weights_loaded());
    EXPECT_TRUE(engine.value.weight_arena_bytes() > 0);
    EXPECT_TRUE(engine.value.workspace_arena_bytes() > 0);
    EXPECT_TRUE(engine.value.config().layers.size() == 2);
    EXPECT_TRUE(!engine.value.config().layers[0].full_attention);
    EXPECT_TRUE(engine.value.config().layers[1].full_attention);
    EXPECT_TRUE(engine.value.config().layers[1].head_dim == 8);
    EXPECT_TRUE(engine.value.config().layers[1].n_kv_heads == 1);

    std::vector<std::vector<uint32_t>> sequences;
    std::vector<float> expected;
    EXPECT_TRUE(load_tokens(dir + "/tokens.txt", sequences));
    EXPECT_TRUE(load_f32(dir + "/expected.f32", expected));
    const uint32_t width = engine.value.config().embedding_dim;
    EXPECT_TRUE(expected.size() == sequences.size() * width);

    size_t offset = 0;
    for (const std::vector<uint32_t>& tokens : sequences) {
        std::vector<float> first(width);
        std::vector<float> second(width);
        Status status = engine.value.embed(tokens, first);
        EXPECT_TRUE(status);
        if (!status) {
            std::cerr << "embed: " << status.message << "\n";
            return;
        }
        status = engine.value.embed(tokens, second);
        EXPECT_TRUE(status);
        for (uint32_t i = 0; i < width; ++i) {
            EXPECT_TRUE(first[i] == second[i]);
            EXPECT_TRUE(std::isfinite(first[i]));
        }

        double norm = 0.0;
        double max_abs = 0.0;
        for (uint32_t i = 0; i < width; ++i) {
            norm += static_cast<double>(first[i]) * static_cast<double>(first[i]);
            const double diff = std::fabs(
                static_cast<double>(first[i]) - static_cast<double>(expected[offset + i]));
            if (diff > max_abs) {
                max_abs = diff;
            }
        }
        EXPECT_TRUE(std::fabs(std::sqrt(norm) - 1.0) < 1e-5);
        const double score = cosine(first.data(), expected.data() + offset, width);
        std::cout << "tiny reference cosine=" << score << " max_abs=" << max_abs << "\n";
        if (!(score >= 0.999999) || max_abs > 2e-4) {
            std::cerr << "reference mismatch cosine=" << score << " max_abs=" << max_abs << "\n";
            std::cerr << "got     ";
            for (float value : first) {
                std::cerr << value << " ";
            }
            std::cerr << "\nexpected ";
            for (uint32_t i = 0; i < width; ++i) {
                std::cerr << expected[offset + i] << " ";
            }
            std::cerr << "\n";
            ++g_failures;
        }
        offset += width;
    }

    std::vector<uint32_t> too_long(9, 1);
    std::vector<float> out(width);
    Status too_long_status = engine.value.embed(too_long, out);
    EXPECT_TRUE(!too_long_status);

    std::vector<uint32_t> bad_token = {100};
    Status bad_token_status = engine.value.embed(bad_token, out);
    EXPECT_TRUE(!bad_token_status);

    std::vector<float> bad_width(3);
    Status bad_width_status = engine.value.embed(sequences[0], bad_width);
    EXPECT_TRUE(!bad_width_status);

    std::vector<float> matryoshka(128);
    Status matryoshka_status = engine.value.embed(sequences[0], matryoshka);
    EXPECT_TRUE(!matryoshka_status);
}

void test_quantize_one(
    EmbeddingGemma2Quant quant,
    const char* name,
    const std::string& dir,
    const std::vector<std::vector<uint32_t>>& sequences,
    const std::vector<float>& expected,
    const std::vector<float>& f32_embeddings,
    uint32_t width) {
    Result<std::unique_ptr<Backend>> backend = create_cpu_backend();
    EXPECT_TRUE(backend.status);
    if (!backend.status) {
        return;
    }
    Result<EmbeddingGemma2Engine> engine = load_embeddinggemma2_from_hf_dir(
        *backend.value,
        dir.c_str(),
        8);
    EXPECT_TRUE(engine.status);
    if (!engine.status) {
        return;
    }

    const size_t workspace_before = engine.value.workspace_arena_bytes();
    const size_t weights_before = engine.value.weight_arena_bytes();
    Status bad = engine.value.quantize_weights(EmbeddingGemma2Quant::f32);
    EXPECT_TRUE(!bad);

    Status status = engine.value.quantize_weights(quant);
    EXPECT_TRUE(status);
    if (!status) {
        std::cerr << name << " quantize: " << status.message << "\n";
        return;
    }
    EXPECT_TRUE(engine.value.weight_quant() == quant);
    EXPECT_TRUE(engine.value.workspace_arena_bytes() == workspace_before);
    std::cout << name
              << " weight_bytes " << weights_before
              << " -> " << engine.value.weight_arena_bytes()
              << " workspace_bytes " << engine.value.workspace_arena_bytes()
              << "\n";

    Status again = engine.value.quantize_weights(quant);
    EXPECT_TRUE(!again);

    size_t offset = 0;
    for (const std::vector<uint32_t>& tokens : sequences) {
        std::vector<float> got(width);
        status = engine.value.embed(tokens, got);
        EXPECT_TRUE(status);
        if (!status) {
            std::cerr << name << " embed: " << status.message << "\n";
            return;
        }
        double norm = 0.0;
        double max_abs_f32 = 0.0;
        for (uint32_t i = 0; i < width; ++i) {
            norm += static_cast<double>(got[i]) * static_cast<double>(got[i]);
            const double diff = std::fabs(
                static_cast<double>(got[i]) - static_cast<double>(f32_embeddings[offset + i]));
            if (diff > max_abs_f32) {
                max_abs_f32 = diff;
            }
            EXPECT_TRUE(std::isfinite(got[i]));
        }
        EXPECT_TRUE(std::fabs(std::sqrt(norm) - 1.0) < 1e-5);
        const double vs_f32 = cosine(got.data(), f32_embeddings.data() + offset, width);
        const double vs_ref = cosine(got.data(), expected.data() + offset, width);
        std::cout << name
                  << " cosine_vs_f32=" << vs_f32
                  << " cosine_vs_reference=" << vs_ref
                  << " max_abs_vs_f32=" << max_abs_f32
                  << "\n";
        // Measured on this Linux machine against the tiny fixture, 2026-10-14,
        // after the quant range includes 0. int8: 0.99984 and 0.999858.
        // int4: 0.920494 and 0.820283. Not the official 32-text gates.
        const double cosine_floor = quant == EmbeddingGemma2Quant::int8 ? 0.9998 : 0.820;
        const double abs_ceil = quant == EmbeddingGemma2Quant::int8 ? 0.014 : 0.39;
        EXPECT_TRUE(vs_f32 >= cosine_floor);
        EXPECT_TRUE(vs_ref >= cosine_floor);
        EXPECT_TRUE(max_abs_f32 <= abs_ceil);
        offset += width;
    }
}

float row_error(const float* got, const float* want, int n) {
    float error = 0.0f;
    for (int i = 0; i < n; ++i) {
        error = std::max(error, std::fabs(got[i] - want[i]));
    }
    return error;
}

void test_same_sign_quant_rows() {
    const float positive[] = {1.0f, 2.0f};
    const float negative[] = {-2.0f, -1.0f};
    const float zeros[] = {0.0f, 0.0f};
    const float constant[] = {1.25f, 1.25f};
    const float mixed[] = {-1.0f, 2.0f};
    float out[2];
    const int bits[] = {8, 4};
    for (int bit : bits) {
        const float bin = bit == 8 ? (2.0f / 255.0f) : (2.0f / 15.0f);
        dequantized_row_for_test(positive, 2, bit, out);
        std::cout << "same-sign +" << bit
                  << " dequant=" << out[0] << "," << out[1] << "\n";
        EXPECT_TRUE(row_error(out, positive, 2) <= bin);
        EXPECT_TRUE(out[1] > out[0] + 0.5f);

        dequantized_row_for_test(negative, 2, bit, out);
        std::cout << "same-sign -" << bit
                  << " dequant=" << out[0] << "," << out[1] << "\n";
        EXPECT_TRUE(row_error(out, negative, 2) <= bin);
        EXPECT_TRUE(out[0] < out[1] - 0.5f);

        dequantized_row_for_test(zeros, 2, bit, out);
        EXPECT_TRUE(out[0] == 0.0f && out[1] == 0.0f);

        dequantized_row_for_test(constant, 2, bit, out);
        EXPECT_TRUE(row_error(out, constant, 2) <= 1e-5f);

        const float mixed_bin = bit == 8 ? (3.0f / 255.0f) : (3.0f / 15.0f);
        dequantized_row_for_test(mixed, 2, bit, out);
        EXPECT_TRUE(row_error(out, mixed, 2) <= mixed_bin);
    }
}

void test_quantize() {
    const std::string dir = fixture_path("embeddinggemma2_tiny");
    Result<std::unique_ptr<Backend>> backend = create_cpu_backend();
    EXPECT_TRUE(backend.status);
    if (!backend.status) {
        return;
    }
    Result<EmbeddingGemma2Engine> f32_engine = load_embeddinggemma2_from_hf_dir(
        *backend.value,
        dir.c_str(),
        8);
    EXPECT_TRUE(f32_engine.status);
    if (!f32_engine.status) {
        return;
    }

    std::vector<std::vector<uint32_t>> sequences;
    std::vector<float> expected;
    EXPECT_TRUE(load_tokens(dir + "/tokens.txt", sequences));
    EXPECT_TRUE(load_f32(dir + "/expected.f32", expected));
    const uint32_t width = f32_engine.value.config().embedding_dim;
    std::vector<float> f32_embeddings(sequences.size() * width);
    size_t offset = 0;
    for (const std::vector<uint32_t>& tokens : sequences) {
        Status status = f32_engine.value.embed(
            tokens,
            std::span<float>(f32_embeddings.data() + offset, width));
        EXPECT_TRUE(status);
        offset += width;
    }

    test_quantize_one(
        EmbeddingGemma2Quant::int8,
        "int8",
        dir,
        sequences,
        expected,
        f32_embeddings,
        width);
    test_quantize_one(
        EmbeddingGemma2Quant::int4,
        "int4",
        dir,
        sequences,
        expected,
        f32_embeddings,
        width);
}

}  // namespace

int main() {
    test_official_config();
    test_config_rejects();
    test_tiny_reference();
    test_same_sign_quant_rows();
    test_quantize();
    if (g_failures != 0) {
        std::cerr << g_failures << " failure(s)\n";
        return 1;
    }
    std::cout << "embedder tests passed\n";
    return 0;
}
