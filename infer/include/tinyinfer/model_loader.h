#pragma once

#include "tinyinfer/llama.h"

namespace tinyinfer {

struct HfModelFiles {
    const char* config_json = nullptr;
    const char* model_safetensors = nullptr;
};

Result<LlamaConfig> load_llama_config_json(const char* path);

Status load_llama_safetensors(
    LlamaInferEngine& engine,
    const char* path);

Result<LlamaInferEngine> load_llama_from_hf_dir(
    Backend& backend,
    const char* model_dir,
    uint32_t max_seq_len);

Result<LlamaInferEngine> load_llama_from_hf_files(
    Backend& backend,
    const HfModelFiles& files,
    uint32_t max_seq_len);

}  // namespace tinyinfer
