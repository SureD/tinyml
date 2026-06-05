#include "tinyinfer/model_loader.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace tinyinfer;

namespace {

struct BenchResult {
    const char* backend = nullptr;
    uint32_t prompt_len = 0;
    uint32_t max_seq_len = 0;
    uint32_t max_new_tokens = 0;
    double prefill_ms = 0.0;
    double decode_total_ms = 0.0;
};

void usage(const char* program) {
    std::cerr
        << "usage: " << program
        << " <model_dir> <backend> <max_seq_len> <max_new_tokens>"
        << " <token_id> [token_id...]\n";
}

bool parse_u32(const char* text, uint32_t& out) {
    if (text == nullptr || text[0] == '\0') {
        return false;
    }

    try {
        size_t parsed_chars = 0;
        const unsigned long value = std::stoul(text, &parsed_chars, 10);
        if (text[parsed_chars] != '\0' ||
            value > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        out = static_cast<uint32_t>(value);
        return true;
    } catch (...) {
        return false;
    }
}

Result<std::unique_ptr<Backend>> create_backend(const std::string& name) {
    if (name == "cpu") {
        return create_cpu_backend();
    }
    if (name == "metal") {
        return create_metal_backend();
    }
    return {
        Status::invalid_argument_status("unsupported backend"),
        nullptr,
    };
}

Status validate_prompt(
    const std::vector<TokenId>& prompt,
    const LlamaConfig& config,
    uint32_t max_seq_len,
    uint32_t max_new_tokens) {
    if (prompt.empty()) {
        return Status::invalid_argument_status("prompt token list must not be empty");
    }
    if (prompt.size() + max_new_tokens > max_seq_len) {
        return Status::invalid_argument_status(
            "prompt plus max_new_tokens exceeds max_seq_len");
    }
    for (TokenId token : prompt) {
        if (token >= config.vocab_size) {
            return Status::invalid_argument_status("prompt token exceeds vocab_size");
        }
    }
    return Status::success();
}

double elapsed_ms(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end) {
    const std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count();
}

Status run_warmup(
    LlamaInferEngine& engine,
    Backend& backend,
    const std::vector<TokenId>& prompt) {
    TokenId token = 0;
    Status status = engine.prefill(prompt, token);
    if (!status) {
        return status;
    }

    status = engine.decode_one(token, token);
    if (!status) {
        return status;
    }

    return backend.synchronize();
}

Status run_measured(
    LlamaInferEngine& engine,
    Backend& backend,
    const char* backend_name,
    const std::vector<TokenId>& prompt,
    uint32_t max_seq_len,
    uint32_t max_new_tokens,
    BenchResult& result) {
    TokenId token = 0;

    const auto prefill_start = std::chrono::steady_clock::now();
    Status status = engine.prefill(prompt, token);
    if (!status) {
        return status;
    }
    status = backend.synchronize();
    if (!status) {
        return status;
    }
    const auto prefill_end = std::chrono::steady_clock::now();

    const auto decode_start = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < max_new_tokens; ++i) {
        status = engine.decode_one(token, token);
        if (!status) {
            return status;
        }
    }
    status = backend.synchronize();
    if (!status) {
        return status;
    }
    const auto decode_end = std::chrono::steady_clock::now();

    result.backend = backend_name;
    result.prompt_len = static_cast<uint32_t>(prompt.size());
    result.max_seq_len = max_seq_len;
    result.max_new_tokens = max_new_tokens;
    result.prefill_ms = elapsed_ms(prefill_start, prefill_end);
    result.decode_total_ms = elapsed_ms(decode_start, decode_end);
    return Status::success();
}

void print_csv(const BenchResult& result) {
    const double decode_ms_per_token =
        result.decode_total_ms / static_cast<double>(result.max_new_tokens);
    const double tokens_per_sec =
        static_cast<double>(result.max_new_tokens) /
        (result.decode_total_ms / 1000.0);

    std::cout
        << "backend,prompt_len,max_seq_len,max_new_tokens,"
        << "prefill_ms,decode_total_ms,decode_ms_per_token,tokens_per_sec\n";
    std::cout
        << result.backend << ","
        << result.prompt_len << ","
        << result.max_seq_len << ","
        << result.max_new_tokens << ","
        << std::fixed << std::setprecision(3)
        << result.prefill_ms << ","
        << result.decode_total_ms << ","
        << decode_ms_per_token << ","
        << tokens_per_sec << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 6) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    const char* model_dir = argv[1];
    const std::string backend_name = argv[2];

    uint32_t max_seq_len = 0;
    uint32_t max_new_tokens = 0;
    if (!parse_u32(argv[3], max_seq_len) ||
        !parse_u32(argv[4], max_new_tokens) ||
        max_seq_len == 0 ||
        max_new_tokens == 0) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    std::vector<TokenId> prompt;
    prompt.reserve(static_cast<size_t>(argc - 5));
    for (int i = 5; i < argc; ++i) {
        uint32_t token = 0;
        if (!parse_u32(argv[i], token)) {
            std::cerr << "invalid token id: " << argv[i] << "\n";
            return EXIT_FAILURE;
        }
        prompt.push_back(token);
    }

    Result<std::unique_ptr<Backend>> backend = create_backend(backend_name);
    if (!backend.status) {
        std::cerr << "create_backend failed: " << backend.status.message << "\n";
        return EXIT_FAILURE;
    }

    Result<LlamaInferEngine> engine =
        load_llama_from_hf_dir(*backend.value, model_dir, max_seq_len);
    if (!engine.status) {
        std::cerr << "load_llama_from_hf_dir failed: "
                  << engine.status.message << "\n";
        return EXIT_FAILURE;
    }

    Status status = validate_prompt(
        prompt,
        engine.value.config(),
        max_seq_len,
        max_new_tokens);
    if (!status) {
        std::cerr << "invalid benchmark input: " << status.message << "\n";
        return EXIT_FAILURE;
    }

    status = run_warmup(engine.value, *backend.value, prompt);
    if (!status) {
        std::cerr << "warmup failed: " << status.message << "\n";
        return EXIT_FAILURE;
    }

    engine.value.reset();

    BenchResult result;
    status = run_measured(
        engine.value,
        *backend.value,
        backend_name.c_str(),
        prompt,
        max_seq_len,
        max_new_tokens,
        result);
    if (!status) {
        std::cerr << "measured run failed: " << status.message << "\n";
        return EXIT_FAILURE;
    }

    print_csv(result);
    return EXIT_SUCCESS;
}
