#include "tinyinfer/embedder.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace tinyinfer;

namespace {

constexpr int kWarmupCalls = 1;
constexpr int kMeasuredCalls = 500;
constexpr uint32_t kMaxSeqLen = 64;

struct Case {
    std::string name;
    std::vector<uint32_t> tokens;
    int reference_index = -1;
};

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
    const std::streamoff size = file.tellg();
    if (size < 0 || static_cast<std::streamoff>(size) % static_cast<std::streamoff>(sizeof(float)) != 0) {
        return false;
    }
    file.seekg(0, std::ios::beg);
    values.resize(static_cast<size_t>(size) / sizeof(float));
    if (!values.empty()) {
        file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(size));
    }
    return static_cast<bool>(file);
}

std::string read_first_value(const char* path, const char* key) {
    std::ifstream file(path);
    if (!file) {
        return "unavailable";
    }
    std::string line;
    const std::string prefix(key);
    while (std::getline(file, line)) {
        if (line.compare(0, prefix.size(), prefix) == 0) {
            std::string value = line.substr(prefix.size());
            const size_t start = value.find_first_not_of(" \t:");
            if (start == std::string::npos) {
                return "unavailable";
            }
            return value.substr(start);
        }
    }
    return "unavailable";
}

bool read_status_kib(const char* key, long& out) {
    std::ifstream file("/proc/self/status");
    if (!file) {
        return false;
    }
    std::string line;
    const std::string prefix(key);
    while (std::getline(file, line)) {
        if (line.compare(0, prefix.size(), prefix) != 0) {
            continue;
        }
        std::istringstream input(line.substr(prefix.size()));
        input >> out;
        return static_cast<bool>(input);
    }
    return false;
}

double percentile(std::vector<double> samples, double p) {
    std::sort(samples.begin(), samples.end());
    const double rank = p * static_cast<double>(samples.size() - 1);
    const size_t index = static_cast<size_t>(std::llround(rank));
    return samples[index];
}

const char* quant_name(EmbeddingGemma2Quant quant) {
    if (quant == EmbeddingGemma2Quant::int8) {
        return "int8";
    }
    if (quant == EmbeddingGemma2Quant::int4) {
        return "int4";
    }
    return "f32";
}

std::vector<uint32_t> patterned_tokens(uint32_t length) {
    std::vector<uint32_t> tokens(length);
    for (uint32_t i = 0; i < length; ++i) {
        tokens[i] = 1 + (i % 19);
    }
    return tokens;
}

bool finite_unit(const std::vector<float>& embedding) {
    double norm = 0.0;
    for (float value : embedding) {
        if (!std::isfinite(value)) {
            return false;
        }
        norm += static_cast<double>(value) * static_cast<double>(value);
    }
    return std::fabs(std::sqrt(norm) - 1.0) < 1e-4;
}

void print_machine() {
    const std::time_t now = std::time(nullptr);
    std::cout << "timestamp " << std::put_time(std::gmtime(&now), "%Y-%m-%dT%H:%M:%SZ") << "\n";
    std::cout << "build_type " << TINYINFER_BENCH_BUILD_TYPE << "\n";
    std::cout << "cpu " << read_first_value("/proc/cpuinfo", "model name") << "\n";
    std::cout << "mem_total " << read_first_value("/proc/meminfo", "MemTotal") << "\n";
    std::cout << "fixture embeddinggemma2_tiny\n";
    std::cout << "backend cpu\n";
    std::cout << "batch 1\n";
    std::cout << "max_seq_len " << kMaxSeqLen << "\n";
    std::cout << "warmup_calls " << kWarmupCalls << "\n";
    std::cout << "measured_calls " << kMeasuredCalls << "\n";
    std::cout << "percentile index=llround(p*(n-1)) on sorted embed() times\n";
    std::cout << "official_model not_run\n";
    std::cout << "cosine_gate_0.999 not_run\n";
    std::cout << "metal not_yet_measured\n";
    std::cout << "macbook_air not_yet_measured\n";
    std::cout << "rtx_3060 not_yet_measured\n";

    Result<std::unique_ptr<Backend>> metal = create_metal_backend();
    std::cout << "create_metal_backend " << (metal.status ? "ok" : metal.status.message) << "\n";
}

Status bench_quant(
    Backend& backend,
    const std::string& dir,
    EmbeddingGemma2Quant quant,
    const std::vector<Case>& cases,
    const std::vector<float>& expected,
    uint32_t width,
    std::vector<std::vector<float>>& f32_embeddings) {
    Result<EmbeddingGemma2Engine> engine = load_embeddinggemma2_from_hf_dir(
        backend,
        dir.c_str(),
        kMaxSeqLen);
    if (!engine.status) {
        std::cerr << quant_name(quant) << " load: " << engine.status.message << "\n";
        return engine.status;
    }
    if (quant != EmbeddingGemma2Quant::f32) {
        Status status = engine.value.quantize_weights(quant);
        if (!status) {
            std::cerr << quant_name(quant) << " quantize: " << status.message << "\n";
            return status;
        }
    }
    if (engine.value.weight_quant() != quant) {
        return Status::backend_error_status("weight packing does not match the bench cell");
    }
    if (engine.value.config().embedding_dim != width) {
        return Status::backend_error_status("fixture width changed");
    }

    std::cout << "quant " << quant_name(quant)
              << " weight_bytes " << engine.value.weight_arena_bytes()
              << " workspace_bytes " << engine.value.workspace_arena_bytes()
              << " embedding_dim " << width
              << "\n";
    std::cout << "case quant seq_len p50_ms p95_ms seq_per_s tok_per_s finite cosine_vs_f32 cosine_vs_expected\n";

    for (size_t case_index = 0; case_index < cases.size(); ++case_index) {
        const Case& item = cases[case_index];
        std::vector<float> embedding(width);
        for (int i = 0; i < kWarmupCalls; ++i) {
            Status status = engine.value.embed(item.tokens, embedding);
            if (!status) {
                std::cerr << item.name << " warmup: " << status.message << "\n";
                return status;
            }
        }

        std::vector<double> samples(static_cast<size_t>(kMeasuredCalls));
        for (int i = 0; i < kMeasuredCalls; ++i) {
            const auto start = std::chrono::steady_clock::now();
            Status status = engine.value.embed(item.tokens, embedding);
            const auto end = std::chrono::steady_clock::now();
            if (!status) {
                std::cerr << item.name << " embed: " << status.message << "\n";
                return status;
            }
            samples[static_cast<size_t>(i)] =
                std::chrono::duration<double, std::milli>(end - start).count();
        }

        if (!finite_unit(embedding)) {
            std::cerr << item.name << " output is not a finite unit vector\n";
            return Status::backend_error_status("bench output is not a finite unit vector");
        }
        if (quant == EmbeddingGemma2Quant::f32) {
            f32_embeddings[case_index] = embedding;
        }

        const double p50 = percentile(samples, 0.50);
        const double p95 = percentile(samples, 0.95);
        if (!(p50 > 0.0)) {
            std::cerr << item.name << " p50 is not positive\n";
            return Status::backend_error_status("bench p50 is not positive");
        }
        const double seq_per_s = 1000.0 / p50;
        const double tok_per_s = seq_per_s * static_cast<double>(item.tokens.size());

        double vs_f32 = 1.0;
        if (quant != EmbeddingGemma2Quant::f32) {
            vs_f32 = cosine(embedding.data(), f32_embeddings[case_index].data(), width);
        }
        std::cout << std::setprecision(9) << std::defaultfloat;
        std::cout << item.name << " " << quant_name(quant) << " " << item.tokens.size()
                  << " " << p50 << " " << p95 << " " << seq_per_s << " " << tok_per_s
                  << " 1 " << vs_f32 << " ";
        if (item.reference_index >= 0) {
            const float* ref = expected.data() + static_cast<size_t>(item.reference_index) * width;
            std::cout << cosine(embedding.data(), ref, width);
        } else {
            std::cout << "na";
        }
        std::cout << "\n";
    }

    long vmrss = -1;
    long vmhwm = -1;
    const bool have_rss = read_status_kib("VmRSS:", vmrss) && read_status_kib("VmHWM:", vmhwm);
    std::cout << "rss_after " << quant_name(quant) << " ";
    if (have_rss) {
        std::cout << "vmrss_kib " << vmrss << " vmhwm_kib " << vmhwm << "\n";
    } else {
        std::cout << "unavailable\n";
    }
    return Status::success();
}

}  // namespace

int main() {
    print_machine();

    const std::string dir = std::string(TINYINFER_EMBEDDER_FIXTURE_DIR) + "/embeddinggemma2_tiny";
    std::vector<std::vector<uint32_t>> fixture_tokens;
    std::vector<float> expected;
    if (!load_tokens(dir + "/tokens.txt", fixture_tokens) || !load_f32(dir + "/expected.f32", expected)) {
        std::cerr << "failed to read the tiny fixture\n";
        return 1;
    }

    std::vector<Case> cases;
    for (size_t i = 0; i < fixture_tokens.size(); ++i) {
        cases.push_back(
            {std::string("fixture") + std::to_string(i), std::move(fixture_tokens[i]), static_cast<int>(i)});
    }
    cases.push_back({"pattern1", patterned_tokens(1), -1});
    cases.push_back({"pattern8", patterned_tokens(8), -1});
    cases.push_back({"pattern32", patterned_tokens(32), -1});
    cases.push_back({"pattern64", patterned_tokens(64), -1});

    Result<std::unique_ptr<Backend>> backend = create_cpu_backend();
    if (!backend.status) {
        std::cerr << "cpu backend: " << backend.status.message << "\n";
        return 1;
    }

    Result<EmbeddingGemma2Config> config = load_embeddinggemma2_config_json((dir + "/config.json").c_str());
    if (!config.status) {
        std::cerr << "config: " << config.status.message << "\n";
        return 1;
    }
    const uint32_t width = config.value.embedding_dim;
    size_t reference_rows = 0;
    for (const Case& item : cases) {
        if (item.reference_index >= 0) {
            ++reference_rows;
        }
    }
    if (expected.size() != reference_rows * width) {
        std::cerr << "expected.f32 does not match the fixture sequences\n";
        return 1;
    }

    std::vector<std::vector<float>> f32_embeddings(cases.size());
    const EmbeddingGemma2Quant quants[] = {
        EmbeddingGemma2Quant::f32,
        EmbeddingGemma2Quant::int8,
        EmbeddingGemma2Quant::int4,
    };
    for (EmbeddingGemma2Quant quant : quants) {
        Status status = bench_quant(
            *backend.value,
            dir,
            quant,
            cases,
            expected,
            width,
            f32_embeddings);
        if (!status) {
            return 1;
        }
    }
    return 0;
}
