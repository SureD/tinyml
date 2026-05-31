#include "tinyinfer/model_loader.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace tinyinfer {
namespace {

using Json = nlohmann::json;

constexpr uint64_t kSafetensorsPrefixBytes = 8;
constexpr uint64_t kMaxSafetensorsHeaderBytes = 64ULL * 1024ULL * 1024ULL;

enum class SafeDType {
    f32,
    bf16,
    unsupported,
};

struct SafeTensorEntry {
    SafeDType dtype = SafeDType::unsupported;
    std::vector<int64_t> shape;
    uint64_t begin = 0;
    uint64_t end = 0;
};

struct SafeTensorsHeader {
    uint64_t file_size = 0;
    uint64_t data_base = 0;
    std::unordered_map<std::string, SafeTensorEntry> tensors;
};

struct ExpectedTensor {
    std::string key;
    TensorView view;
};

bool checked_add(uint64_t a, uint64_t b, uint64_t& out) {
    if (a > std::numeric_limits<uint64_t>::max() - b) {
        return false;
    }
    out = a + b;
    return true;
}

bool checked_mul(uint64_t a, uint64_t b, uint64_t& out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

bool path_join(const char* dir, const char* file, std::string& out) {
    if (dir == nullptr || file == nullptr || dir[0] == '\0' || file[0] == '\0') {
        return false;
    }

    out = dir;
    if (!out.empty() && out.back() != '/') {
        out.push_back('/');
    }
    out += file;
    return true;
}

Result<std::string> read_text_file(const char* path) {
    if (path == nullptr || path[0] == '\0') {
        return {Status::invalid_argument_status("file path is empty"), {}};
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return {Status::invalid_argument_status("failed to open file"), {}};
    }

    file.seekg(0, std::ios::end);
    const std::streampos size_pos = file.tellg();
    if (size_pos < 0) {
        return {Status::invalid_argument_status("failed to read file size"), {}};
    }
    file.seekg(0, std::ios::beg);

    std::string data(static_cast<size_t>(size_pos), '\0');
    if (!data.empty()) {
        file.read(data.data(), static_cast<std::streamsize>(data.size()));
        if (!file) {
            return {Status::invalid_argument_status("failed to read file"), {}};
        }
    }

    return {Status::success(), std::move(data)};
}

Result<Json> parse_json_text(const std::string& text) {
    Json parsed = Json::parse(text, nullptr, false);
    if (parsed.is_discarded()) {
        return {Status::invalid_argument_status("failed to parse JSON"), {}};
    }
    return {Status::success(), std::move(parsed)};
}

const Json* find_field(const Json& object, const char* key) {
    if (!object.is_object()) {
        return nullptr;
    }
    const auto it = object.find(key);
    return it == object.end() ? nullptr : &(*it);
}

bool json_to_u64(const Json& value, uint64_t& out) {
    if (value.is_number_unsigned()) {
        out = value.get<uint64_t>();
        return true;
    }
    if (value.is_number_integer()) {
        const int64_t signed_value = value.get<int64_t>();
        if (signed_value < 0) {
            return false;
        }
        out = static_cast<uint64_t>(signed_value);
        return true;
    }
    return false;
}

Status require_u32(const Json& object, const char* key, uint32_t& out) {
    const Json* value = find_field(object, key);
    if (value == nullptr) {
        return Status::invalid_config_status("missing required config field");
    }

    uint64_t parsed = 0;
    if (!json_to_u64(*value, parsed) || parsed > std::numeric_limits<uint32_t>::max()) {
        return Status::invalid_config_status("invalid integer config field");
    }

    out = static_cast<uint32_t>(parsed);
    return Status::success();
}

Status optional_float(
    const Json& object,
    const char* key,
    float fallback,
    float& out) {
    const Json* value = find_field(object, key);
    if (value == nullptr || value->is_null()) {
        out = fallback;
        return Status::success();
    }
    if (!value->is_number()) {
        return Status::invalid_config_status("invalid float config field");
    }

    const double parsed = value->get<double>();
    if (!std::isfinite(parsed) ||
        parsed < -std::numeric_limits<float>::max() ||
        parsed > std::numeric_limits<float>::max()) {
        return Status::invalid_config_status("float config field is out of range");
    }

    out = static_cast<float>(parsed);
    return Status::success();
}

Status require_string_value(const Json& object, const char* key, const char* expected) {
    const Json* value = find_field(object, key);
    if (value == nullptr) {
        return Status::success();
    }
    if (!value->is_string() || value->get<std::string>() != expected) {
        return Status::invalid_config_status("unsupported string config value");
    }
    return Status::success();
}

Status require_bool_value(const Json& object, const char* key, bool expected) {
    const Json* value = find_field(object, key);
    if (value == nullptr) {
        return Status::success();
    }
    if (!value->is_boolean() || value->get<bool>() != expected) {
        return Status::invalid_config_status("unsupported bool config value");
    }
    return Status::success();
}

Status require_u32_value(const Json& object, const char* key, uint32_t expected) {
    const Json* value = find_field(object, key);
    if (value == nullptr) {
        return Status::success();
    }

    uint64_t parsed = 0;
    if (!json_to_u64(*value, parsed) || parsed != expected) {
        return Status::invalid_config_status("unsupported integer config value");
    }
    return Status::success();
}

Status require_null_or_absent(const Json& object, const char* key) {
    const Json* value = find_field(object, key);
    if (value == nullptr || value->is_null()) {
        return Status::success();
    }
    return Status::invalid_config_status("unsupported non-null config value");
}

SafeDType parse_safe_dtype(const std::string& dtype) {
    if (dtype == "F32") {
        return SafeDType::f32;
    }
    if (dtype == "BF16") {
        return SafeDType::bf16;
    }
    return SafeDType::unsupported;
}

uint64_t safe_dtype_size(SafeDType dtype) {
    switch (dtype) {
    case SafeDType::f32:
        return 4;
    case SafeDType::bf16:
        return 2;
    case SafeDType::unsupported:
        return 0;
    }
    return 0;
}

Status shape_numel(const std::vector<int64_t>& shape, uint64_t& out) {
    uint64_t total = 1;
    for (int64_t dim : shape) {
        if (dim < 0) {
            return Status::invalid_argument_status("safetensors shape has negative dim");
        }
        uint64_t next = 0;
        if (!checked_mul(total, static_cast<uint64_t>(dim), next)) {
            return Status::invalid_argument_status("safetensors shape size overflow");
        }
        total = next;
    }

    out = total;
    return Status::success();
}

uint64_t read_le_u64(const std::array<unsigned char, 8>& bytes) {
    uint64_t value = 0;
    for (uint32_t i = 0; i < bytes.size(); ++i) {
        value |= static_cast<uint64_t>(bytes[i]) << (i * 8);
    }
    return value;
}

bool read_exact_at(std::ifstream& file, uint64_t offset, void* dst, size_t bytes) {
    if (bytes == 0) {
        return true;
    }
    if (dst == nullptr || offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        return false;
    }

    file.clear();
    file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!file) {
        return false;
    }
    file.read(static_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    return static_cast<size_t>(file.gcount()) == bytes && static_cast<bool>(file);
}

Status parse_safe_shape(const Json& value, std::vector<int64_t>& out) {
    if (!value.is_array()) {
        return Status::invalid_argument_status("safetensors shape must be an array");
    }

    out.clear();
    out.reserve(value.size());
    for (const Json& dim_value : value) {
        uint64_t dim = 0;
        if (!json_to_u64(dim_value, dim) ||
            dim > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            return Status::invalid_argument_status("invalid safetensors shape dim");
        }
        out.push_back(static_cast<int64_t>(dim));
    }

    return Status::success();
}

Status parse_safe_offsets(const Json& value, uint64_t& begin, uint64_t& end) {
    if (!value.is_array() || value.size() != 2) {
        return Status::invalid_argument_status("safetensors data_offsets must have two values");
    }
    if (!json_to_u64(value[0], begin) || !json_to_u64(value[1], end) || begin > end) {
        return Status::invalid_argument_status("invalid safetensors data_offsets");
    }
    return Status::success();
}

Status parse_safe_tensor_entry(
    const Json& value,
    uint64_t data_base,
    uint64_t file_size,
    SafeTensorEntry& out) {
    if (!value.is_object()) {
        return Status::invalid_argument_status("safetensors tensor entry must be an object");
    }

    const Json* dtype_value = find_field(value, "dtype");
    const Json* shape_value = find_field(value, "shape");
    const Json* offsets_value = find_field(value, "data_offsets");
    if (dtype_value == nullptr || shape_value == nullptr || offsets_value == nullptr) {
        return Status::invalid_argument_status("safetensors tensor entry is missing fields");
    }
    if (!dtype_value->is_string()) {
        return Status::invalid_argument_status("safetensors dtype must be a string");
    }

    out.dtype = parse_safe_dtype(dtype_value->get<std::string>());

    Status status = parse_safe_shape(*shape_value, out.shape);
    if (!status) {
        return status;
    }
    status = parse_safe_offsets(*offsets_value, out.begin, out.end);
    if (!status) {
        return status;
    }

    uint64_t file_end = 0;
    if (!checked_add(data_base, out.end, file_end) || file_end > file_size) {
        return Status::invalid_argument_status("safetensors tensor data exceeds file size");
    }

    const uint64_t item_size = safe_dtype_size(out.dtype);
    if (item_size != 0) {
        uint64_t numel = 0;
        status = shape_numel(out.shape, numel);
        if (!status) {
            return status;
        }
        uint64_t expected_bytes = 0;
        if (!checked_mul(numel, item_size, expected_bytes)) {
            return Status::invalid_argument_status("safetensors tensor byte size overflow");
        }
        if (out.end - out.begin != expected_bytes) {
            return Status::invalid_argument_status("safetensors tensor byte size mismatch");
        }
    }

    return Status::success();
}

Result<SafeTensorsHeader> parse_safetensors_header(const char* path) {
    if (path == nullptr || path[0] == '\0') {
        return {Status::invalid_argument_status("safetensors path is empty"), {}};
    }

    std::error_code ec;
    const uintmax_t file_bytes = std::filesystem::file_size(path, ec);
    if (ec || file_bytes < kSafetensorsPrefixBytes ||
        file_bytes > std::numeric_limits<uint64_t>::max()) {
        return {Status::invalid_argument_status("invalid safetensors file size"), {}};
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return {Status::invalid_argument_status("failed to open safetensors file"), {}};
    }

    std::array<unsigned char, 8> prefix = {};
    file.read(reinterpret_cast<char*>(prefix.data()), static_cast<std::streamsize>(prefix.size()));
    if (!file) {
        return {Status::invalid_argument_status("failed to read safetensors header length"), {}};
    }

    SafeTensorsHeader header;
    header.file_size = static_cast<uint64_t>(file_bytes);

    const uint64_t header_len = read_le_u64(prefix);
    if (header_len == 0 || header_len > kMaxSafetensorsHeaderBytes) {
        return {Status::invalid_argument_status("invalid safetensors header length"), {}};
    }
    if (header_len > header.file_size - kSafetensorsPrefixBytes) {
        return {Status::invalid_argument_status("safetensors header exceeds file size"), {}};
    }
    if (header_len > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return {Status::invalid_argument_status("safetensors header is too large"), {}};
    }

    if (!checked_add(kSafetensorsPrefixBytes, header_len, header.data_base)) {
        return {Status::invalid_argument_status("safetensors data base overflow"), {}};
    }

    std::string header_json(static_cast<size_t>(header_len), '\0');
    if (!read_exact_at(file, kSafetensorsPrefixBytes, header_json.data(), header_json.size())) {
        return {Status::invalid_argument_status("failed to read safetensors header"), {}};
    }

    Result<Json> parsed_json = parse_json_text(header_json);
    if (!parsed_json.status) {
        return {parsed_json.status, {}};
    }
    if (!parsed_json.value.is_object()) {
        return {Status::invalid_argument_status("safetensors header must be an object"), {}};
    }

    for (auto it = parsed_json.value.begin(); it != parsed_json.value.end(); ++it) {
        if (it.key() == "__metadata__") {
            continue;
        }

        SafeTensorEntry entry;
        Status status = parse_safe_tensor_entry(
            it.value(),
            header.data_base,
            header.file_size,
            entry);
        if (!status) {
            return {status, {}};
        }
        header.tensors.emplace(it.key(), std::move(entry));
    }

    return {Status::success(), std::move(header)};
}

void add_expected(
    std::vector<ExpectedTensor>& expected,
    const char* key,
    const TensorView& view) {
    expected.push_back({key, view});
}

void add_expected(
    std::vector<ExpectedTensor>& expected,
    const std::string& key,
    const TensorView& view) {
    expected.push_back({key, view});
}

bool is_allowed_extra_key(const std::string& key) {
    return key.ends_with(".self_attn.rotary_emb.inv_freq");
}

bool tensor_shape_matches(const TensorView& view, const SafeTensorEntry& entry) {
    if (view.shape.ndim != entry.shape.size()) {
        return false;
    }
    for (uint32_t i = 0; i < view.shape.ndim; ++i) {
        if (view.dim(i) != entry.shape[i]) {
            return false;
        }
    }
    return true;
}

Status validate_expected_entry(
    const ExpectedTensor& expected,
    const SafeTensorEntry& entry) {
    if (!expected.view.defined()) {
        return Status::invalid_argument_status("destination tensor is not defined");
    }
    if (!expected.view.contiguous() || expected.view.dtype != DType::f32) {
        return Status::invalid_argument_status("destination tensor must be contiguous f32");
    }
    if (!tensor_shape_matches(expected.view, entry)) {
        return Status::invalid_argument_status("checkpoint tensor shape mismatch");
    }
    if (entry.dtype == SafeDType::unsupported) {
        return Status::unimplemented_status("unsupported checkpoint dtype");
    }
    return Status::success();
}

Result<std::vector<float>> read_tensor_as_f32(
    std::ifstream& file,
    const SafeTensorsHeader& header,
    const SafeTensorEntry& entry) {
    uint64_t numel = 0;
    Status status = shape_numel(entry.shape, numel);
    if (!status) {
        return {status, {}};
    }
    if (numel > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return {Status::invalid_argument_status("tensor is too large to load"), {}};
    }

    const size_t count = static_cast<size_t>(numel);
    std::vector<float> out(count);

    uint64_t file_begin = 0;
    if (!checked_add(header.data_base, entry.begin, file_begin)) {
        return {Status::invalid_argument_status("tensor file offset overflow"), {}};
    }

    if (entry.dtype == SafeDType::f32) {
        const uint64_t bytes = entry.end - entry.begin;
        if (bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            return {Status::invalid_argument_status("tensor byte size is too large"), {}};
        }
        if (!read_exact_at(file, file_begin, out.data(), static_cast<size_t>(bytes))) {
            return {Status::invalid_argument_status("failed to read tensor data"), {}};
        }
        return {Status::success(), std::move(out)};
    }

    if (entry.dtype == SafeDType::bf16) {
        std::vector<uint16_t> bf16(count);
        const uint64_t bytes = entry.end - entry.begin;
        if (bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            return {Status::invalid_argument_status("tensor byte size is too large"), {}};
        }
        if (!read_exact_at(file, file_begin, bf16.data(), static_cast<size_t>(bytes))) {
            return {Status::invalid_argument_status("failed to read tensor data"), {}};
        }

        for (size_t i = 0; i < count; ++i) {
            const uint32_t bits = static_cast<uint32_t>(bf16[i]) << 16;
            std::memcpy(&out[i], &bits, sizeof(float));
        }
        return {Status::success(), std::move(out)};
    }

    return {Status::unimplemented_status("unsupported checkpoint dtype"), {}};
}

Status validate_no_unexpected_tensors(
    const SafeTensorsHeader& header,
    const std::unordered_set<std::string>& expected_keys) {
    for (const auto& item : header.tensors) {
        if (expected_keys.contains(item.first) || is_allowed_extra_key(item.first)) {
            continue;
        }
        return Status::invalid_argument_status("unexpected checkpoint tensor");
    }
    return Status::success();
}

}  // namespace

Result<LlamaConfig> load_llama_config_json(const char* path) {
    Result<std::string> text = read_text_file(path);
    if (!text.status) {
        return {text.status, {}};
    }

    Result<Json> parsed = parse_json_text(text.value);
    if (!parsed.status) {
        return {parsed.status, {}};
    }
    if (!parsed.value.is_object()) {
        return {Status::invalid_config_status("config JSON must be an object"), {}};
    }

    Status status = require_string_value(parsed.value, "model_type", "llama");
    if (!status) {
        return {status, {}};
    }
    status = require_string_value(parsed.value, "hidden_act", "silu");
    if (!status) {
        return {status, {}};
    }
    status = require_bool_value(parsed.value, "attention_bias", false);
    if (!status) {
        return {status, {}};
    }
    status = require_bool_value(parsed.value, "tie_word_embeddings", false);
    if (!status) {
        return {status, {}};
    }
    status = require_u32_value(parsed.value, "pretraining_tp", 1);
    if (!status) {
        return {status, {}};
    }
    status = require_null_or_absent(parsed.value, "rope_scaling");
    if (!status) {
        return {status, {}};
    }

    LlamaConfig config;
    status = require_u32(parsed.value, "num_hidden_layers", config.n_layers);
    if (!status) {
        return {status, {}};
    }
    status = require_u32(parsed.value, "hidden_size", config.hidden_size);
    if (!status) {
        return {status, {}};
    }
    status = require_u32(parsed.value, "intermediate_size", config.intermediate_size);
    if (!status) {
        return {status, {}};
    }
    status = require_u32(parsed.value, "num_attention_heads", config.n_heads);
    if (!status) {
        return {status, {}};
    }
    status = require_u32(parsed.value, "num_key_value_heads", config.n_kv_heads);
    if (!status) {
        return {status, {}};
    }
    status = require_u32(parsed.value, "vocab_size", config.vocab_size);
    if (!status) {
        return {status, {}};
    }
    status = require_u32(parsed.value, "max_position_embeddings", config.max_seq_len);
    if (!status) {
        return {status, {}};
    }
    status = optional_float(parsed.value, "rms_norm_eps", 1e-5f, config.rms_eps);
    if (!status) {
        return {status, {}};
    }
    status = optional_float(parsed.value, "rope_theta", 10000.0f, config.rope_theta);
    if (!status) {
        return {status, {}};
    }

    status = config.validate();
    if (!status) {
        return {status, {}};
    }
    return {Status::success(), config};
}

Status load_llama_safetensors(
    LlamaInferEngine& engine,
    const char* path) {
    Result<SafeTensorsHeader> header = parse_safetensors_header(path);
    if (!header.status) {
        return header.status;
    }

    std::vector<ExpectedTensor> expected;
    expected.reserve(3 + static_cast<size_t>(engine.config_.n_layers) * 9);
    add_expected(expected, "model.embed_tokens.weight", engine.model_.token_embedding);
    add_expected(expected, "model.norm.weight", engine.model_.final_norm);
    add_expected(expected, "lm_head.weight", engine.model_.lm_head);

    for (uint32_t i = 0; i < engine.config_.n_layers; ++i) {
        const std::string prefix = "model.layers." + std::to_string(i);
        const LlamaInferEngine::LayerWeights& layer = engine.model_.layers[i];
        add_expected(expected, prefix + ".input_layernorm.weight", layer.attn_norm);
        add_expected(expected, prefix + ".self_attn.q_proj.weight", layer.q_proj);
        add_expected(expected, prefix + ".self_attn.k_proj.weight", layer.k_proj);
        add_expected(expected, prefix + ".self_attn.v_proj.weight", layer.v_proj);
        add_expected(expected, prefix + ".self_attn.o_proj.weight", layer.o_proj);
        add_expected(expected, prefix + ".post_attention_layernorm.weight", layer.ffn_norm);
        add_expected(expected, prefix + ".mlp.gate_proj.weight", layer.gate_proj);
        add_expected(expected, prefix + ".mlp.up_proj.weight", layer.up_proj);
        add_expected(expected, prefix + ".mlp.down_proj.weight", layer.down_proj);
    }
    std::unordered_set<std::string> expected_keys;
    expected_keys.reserve(expected.size());
    for (const ExpectedTensor& item : expected) {
        expected_keys.insert(item.key);
    }

    Status status = validate_no_unexpected_tensors(header.value, expected_keys);
    if (!status) {
        return status;
    }

    for (const ExpectedTensor& item : expected) {
        const auto found = header.value.tensors.find(item.key);
        if (found == header.value.tensors.end()) {
            return Status::invalid_argument_status("missing checkpoint tensor");
        }
        status = validate_expected_entry(item, found->second);
        if (!status) {
            return status;
        }
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return Status::invalid_argument_status("failed to open safetensors file");
    }

    for (const ExpectedTensor& item : expected) {
        const SafeTensorEntry& entry = header.value.tensors.at(item.key);
        Result<std::vector<float>> tensor = read_tensor_as_f32(file, header.value, entry);
        if (!tensor.status) {
            return tensor.status;
        }

        status = engine.backend_->copy_from_host(
            item.view,
            tensor.value.data(),
            item.view.logical_nbytes());
        if (!status) {
            return status;
        }
    }

    return engine.backend_->synchronize();
}

Result<LlamaInferEngine> load_llama_from_hf_files(
    Backend& backend,
    const HfModelFiles& files,
    uint32_t max_seq_len) {
    if (files.config_json == nullptr || files.model_safetensors == nullptr) {
        return {Status::invalid_argument_status("missing Hugging Face model file paths"), {}};
    }

    Result<LlamaConfig> config = load_llama_config_json(files.config_json);
    if (!config.status) {
        return {config.status, {}};
    }

    Result<LlamaInferEngine> engine =
        LlamaInferEngine::create(backend, config.value, max_seq_len);
    if (!engine.status) {
        return engine;
    }

    Status status = load_llama_safetensors(engine.value, files.model_safetensors);
    if (!status) {
        return {status, {}};
    }

    return {Status::success(), std::move(engine.value)};
}

Result<LlamaInferEngine> load_llama_from_hf_dir(
    Backend& backend,
    const char* model_dir,
    uint32_t max_seq_len) {
    std::string config_path;
    std::string safetensors_path;
    if (!path_join(model_dir, "config.json", config_path) ||
        !path_join(model_dir, "model.safetensors", safetensors_path)) {
        return {Status::invalid_argument_status("model directory path is empty"), {}};
    }

    HfModelFiles files;
    files.config_json = config_path.c_str();
    files.model_safetensors = safetensors_path.c_str();
    return load_llama_from_hf_files(backend, files, max_seq_len);
}

}  // namespace tinyinfer
