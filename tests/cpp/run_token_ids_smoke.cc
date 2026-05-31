#include "tinyinfer/model_loader.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace tinyinfer;

namespace {

void usage(const char* program) {
    std::cerr
        << "usage: " << program
        << " <model_dir> <max_seq_len> <max_new_tokens> <token_id> [token_id...]\n";
}

bool parse_u32(const char* text, uint32_t& out) {
    if (text == nullptr || text[0] == '\0') {
        return false;
    }

    try {
        size_t parsed_chars = 0;
        const unsigned long value = std::stoul(text, &parsed_chars, 10);
        if (text[parsed_chars] != '\0' || value > UINT32_MAX) {
            return false;
        }
        out = static_cast<uint32_t>(value);
        return true;
    } catch (...) {
        return false;
    }
}

void print_token_line(const char* label, const std::vector<TokenId>& tokens) {
    std::cout << label << ":";
    for (TokenId token : tokens) {
        std::cout << " " << token;
    }
    std::cout << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    const char* model_dir = argv[1];
    uint32_t max_seq_len = 0;
    uint32_t max_new_tokens = 0;
    if (!parse_u32(argv[2], max_seq_len) ||
        !parse_u32(argv[3], max_new_tokens) ||
        max_seq_len == 0 ||
        max_new_tokens == 0) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    std::vector<TokenId> prompt;
    prompt.reserve(static_cast<size_t>(argc - 4));
    for (int i = 4; i < argc; ++i) {
        uint32_t token = 0;
        if (!parse_u32(argv[i], token)) {
            std::cerr << "invalid token id: " << argv[i] << "\n";
            return EXIT_FAILURE;
        }
        prompt.push_back(token);
    }
    if (prompt.empty()) {
        std::cerr << "prompt token list must not be empty\n";
        return EXIT_FAILURE;
    }
    if (prompt.size() + max_new_tokens > max_seq_len) {
        std::cerr << "prompt plus max_new_tokens exceeds max_seq_len\n";
        return EXIT_FAILURE;
    }

    Result<std::unique_ptr<Backend>> backend = create_cpu_backend();
    if (!backend.status) {
        std::cerr << "create_cpu_backend failed: " << backend.status.message << "\n";
        return EXIT_FAILURE;
    }

    const auto load_start = std::chrono::steady_clock::now();
    Result<LlamaInferEngine> engine =
        load_llama_from_hf_dir(*backend.value, model_dir, max_seq_len);
    const auto load_end = std::chrono::steady_clock::now();
    if (!engine.status) {
        std::cerr << "load_llama_from_hf_dir failed: " << engine.status.message << "\n";
        return EXIT_FAILURE;
    }

    const LlamaConfig& config = engine.value.config();
    for (TokenId token : prompt) {
        if (token >= config.vocab_size) {
            std::cerr << "prompt token exceeds vocab_size: " << token << "\n";
            return EXIT_FAILURE;
        }
    }

    std::vector<TokenId> output(prompt.size() + max_new_tokens, 0);
    uint32_t output_count = 0;
    GenerateConfig generate_config;
    generate_config.max_new_tokens = max_new_tokens;
    generate_config.eos_token_id = 2;
    generate_config.stop_on_eos = true;

    const auto infer_start = std::chrono::steady_clock::now();
    Status status = engine.value.generate(
        prompt,
        output,
        generate_config,
        output_count);
    const auto infer_end = std::chrono::steady_clock::now();
    if (!status) {
        std::cerr << "generate failed: " << status.message << "\n";
        return EXIT_FAILURE;
    }

    std::vector<TokenId> all_tokens(output.begin(), output.begin() + output_count);
    std::vector<TokenId> generated;
    if (output_count > prompt.size()) {
        generated.assign(
            output.begin() + static_cast<std::ptrdiff_t>(prompt.size()),
            output.begin() + output_count);
    }

    for (TokenId token : generated) {
        if (token >= config.vocab_size) {
            std::cerr << "generated token exceeds vocab_size: " << token << "\n";
            return EXIT_FAILURE;
        }
    }

    const std::chrono::duration<double> load_elapsed = load_end - load_start;
    const std::chrono::duration<double> infer_elapsed = infer_end - infer_start;

    std::cout << "token-id smoke passed\n";
    std::cout << "model_dir: " << model_dir << "\n";
    std::cout << "layers: " << config.n_layers << "\n";
    std::cout << "hidden_size: " << config.hidden_size << "\n";
    std::cout << "vocab_size: " << config.vocab_size << "\n";
    std::cout << "max_seq_len: " << engine.value.max_seq_len() << "\n";
    std::cout << "requested_max_new_tokens: " << max_new_tokens << "\n";
    std::cout << "output_count: " << output_count << "\n";
    print_token_line("prompt_tokens", prompt);
    print_token_line("generated_tokens", generated);
    print_token_line("all_tokens", all_tokens);
    std::cout << "load_elapsed_sec: " << load_elapsed.count() << "\n";
    std::cout << "infer_elapsed_sec: " << infer_elapsed.count() << "\n";
    return EXIT_SUCCESS;
}
