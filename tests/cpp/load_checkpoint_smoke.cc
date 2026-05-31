#include "tinyinfer/model_loader.h"

#include <chrono>
#include <cstdint>
#include <iostream>

using namespace tinyinfer;

int main(int argc, char** argv) {
    const char* model_dir = argc >= 2 ? argv[1] : "models/TinyLlama-1.1B-Chat-v1.0";
    const uint32_t max_seq_len = argc >= 3 ? static_cast<uint32_t>(std::stoul(argv[2])) : 2048;

    Result<std::unique_ptr<Backend>> backend = create_cpu_backend();
    if (!backend.status) {
        std::cerr << "create_cpu_backend failed: " << backend.status.message << "\n";
        return 1;
    }

    const auto start = std::chrono::steady_clock::now();
    Result<LlamaInferEngine> engine =
        load_llama_from_hf_dir(*backend.value, model_dir, max_seq_len);
    const auto end = std::chrono::steady_clock::now();
    if (!engine.status) {
        std::cerr << "load_llama_from_hf_dir failed: " << engine.status.message << "\n";
        return 1;
    }

    const std::chrono::duration<double> elapsed = end - start;
    const LlamaConfig& config = engine.value.config();
    std::cout << "loaded checkpoint\n";
    std::cout << "model_dir: " << model_dir << "\n";
    std::cout << "layers: " << config.n_layers << "\n";
    std::cout << "hidden_size: " << config.hidden_size << "\n";
    std::cout << "intermediate_size: " << config.intermediate_size << "\n";
    std::cout << "heads: " << config.n_heads << "\n";
    std::cout << "kv_heads: " << config.n_kv_heads << "\n";
    std::cout << "vocab_size: " << config.vocab_size << "\n";
    std::cout << "max_seq_len: " << engine.value.max_seq_len() << "\n";
    std::cout << "elapsed_sec: " << elapsed.count() << "\n";
    return 0;
}
