#include "tinyinfer/embedder.h"

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
    TensorView* view = nullptr;
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

Status require_positive_f32(const Json& object, const char* key, float& out) {
    const Json* value = find_field(object, key);
    if (value == nullptr || !value->is_number()) {
        return Status::invalid_config_status("missing required float config field");
    }
    const double parsed = value->get<double>();
    if (!std::isfinite(parsed) || parsed <= 0.0 ||
        parsed > static_cast<double>(std::numeric_limits<float>::max())) {
        return Status::invalid_config_status("invalid float config field");
    }
    out = static_cast<float>(parsed);
    return Status::success();
}

Status require_string(const Json& object, const char* key, const char* expected) {
    const Json* value = find_field(object, key);
    if (value == nullptr || !value->is_string() || value->get<std::string>() != expected) {
        return Status::invalid_config_status("unsupported string config value");
    }
    return Status::success();
}

Status optional_false(const Json& object, const char* key) {
    const Json* value = find_field(object, key);
    if (value == nullptr) {
        return Status::success();
    }
    if (!value->is_boolean() || value->get<bool>()) {
        return Status::invalid_config_status("unsupported bool config value");
    }
    return Status::success();
}

Status reject_if_present(const Json& object, const char* key) {
    if (find_field(object, key) != nullptr) {
        return Status::invalid_config_status("unsupported config field");
    }
    return Status::success();
}

Status parse_layer_index(const std::string& key, uint32_t n_layers, uint32_t& out) {
    if (key.empty()) {
        return Status::invalid_config_status("invalid per-layer config key");
    }
    uint32_t value = 0;
    for (char c : key) {
        if (c < '0' || c > '9') {
            return Status::invalid_config_status("invalid per-layer config key");
        }
        const uint32_t digit = static_cast<uint32_t>(c - '0');
        if (value > (std::numeric_limits<uint32_t>::max() - digit) / 10) {
            return Status::invalid_config_status("invalid per-layer config key");
        }
        value = value * 10 + digit;
    }
    if (value >= n_layers) {
        return Status::invalid_config_status("per-layer config index is out of range");
    }
    out = value;
    return Status::success();
}

Status parse_text_config(const Json& text, EmbeddingGemma2Config& config) {
    if (!text.is_object()) {
        return Status::invalid_config_status("text config must be an object");
    }

    Status status = require_string(text, "model_type", "embedding_gemma2_text");
    if (!status) {
        return status;
    }
    status = require_string(text, "hidden_activation", "gelu_pytorch_tanh");
    if (!status) {
        return status;
    }
    status = optional_false(text, "attention_bias");
    if (!status) {
        return status;
    }
    const Json* dtype = find_field(text, "dtype");
    if (dtype != nullptr) {
        if (!dtype->is_string()) {
            return Status::invalid_config_status("invalid dtype");
        }
        const std::string dtype_name = dtype->get<std::string>();
        if (dtype_name == "float16") {
            return Status::invalid_config_status("float16 is not a supported compute or storage dtype");
        }
        if (dtype_name != "bfloat16" && dtype_name != "float32") {
            return Status::invalid_config_status("unsupported storage dtype");
        }
    }
    status = reject_if_present(text, "query_pre_attn_scalar");
    if (!status) {
        return status;
    }
    status = reject_if_present(text, "attn_logit_softcapping");
    if (!status) {
        return status;
    }

    status = require_u32(text, "num_hidden_layers", config.n_layers);
    if (!status) {
        return status;
    }
    status = require_u32(text, "hidden_size", config.hidden_size);
    if (!status) {
        return status;
    }
    status = require_u32(text, "intermediate_size", config.intermediate_size);
    if (!status) {
        return status;
    }
    status = require_u32(text, "num_attention_heads", config.n_heads);
    if (!status) {
        return status;
    }
    status = require_u32(text, "vocab_size", config.vocab_size);
    if (!status) {
        return status;
    }
    status = require_u32(text, "embedding_dim", config.embedding_dim);
    if (!status) {
        return status;
    }
    status = require_u32(text, "hidden_size_per_layer_input", config.ple_dim);
    if (!status) {
        return status;
    }
    status = require_u32(text, "sliding_window", config.sliding_window);
    if (!status) {
        return status;
    }
    status = require_u32(text, "max_position_embeddings", config.max_position_embeddings);
    if (!status) {
        return status;
    }
    status = require_positive_f32(text, "rms_norm_eps", config.rms_eps);
    if (!status) {
        return status;
    }

    uint32_t default_head_dim = 0;
    uint32_t default_kv_heads = 0;
    status = require_u32(text, "head_dim", default_head_dim);
    if (!status) {
        return status;
    }
    status = require_u32(text, "num_key_value_heads", default_kv_heads);
    if (!status) {
        return status;
    }

    const Json* layer_types = find_field(text, "layer_types");
    if (layer_types == nullptr || !layer_types->is_array() || layer_types->size() != config.n_layers) {
        return Status::invalid_config_status("layer_types must list every layer");
    }

    const Json* rope = find_field(text, "rope_parameters");
    if (rope == nullptr || !rope->is_object()) {
        return Status::invalid_config_status("missing rope parameters");
    }
    float sliding_theta = 0.0f;
    float full_theta = 0.0f;
    const Json* sliding_rope = find_field(*rope, "sliding_attention");
    const Json* full_rope = find_field(*rope, "full_attention");
    if (sliding_rope == nullptr || full_rope == nullptr) {
        return Status::invalid_config_status("missing per-layer rope parameters");
    }
    status = require_string(*sliding_rope, "rope_type", "default");
    if (!status) {
        return status;
    }
    status = require_string(*full_rope, "rope_type", "default");
    if (!status) {
        return status;
    }
    status = require_positive_f32(*sliding_rope, "rope_theta", sliding_theta);
    if (!status) {
        return status;
    }
    status = require_positive_f32(*full_rope, "rope_theta", full_theta);
    if (!status) {
        return status;
    }
    if (sliding_rope->size() != 2 || full_rope->size() != 2) {
        return Status::invalid_config_status("unsupported rope parameter");
    }

    config.layers.assign(config.n_layers, {});
    for (uint32_t i = 0; i < config.n_layers; ++i) {
        const Json& type_value = (*layer_types)[i];
        if (!type_value.is_string()) {
            return Status::invalid_config_status("layer type must be a string");
        }
        const std::string type_name = type_value.get<std::string>();
        EmbeddingGemma2Layer& layer = config.layers[i];
        layer.head_dim = default_head_dim;
        layer.n_kv_heads = default_kv_heads;
        if (type_name == "sliding_attention") {
            layer.full_attention = false;
            layer.rope_theta = sliding_theta;
        } else if (type_name == "full_attention") {
            layer.full_attention = true;
            layer.rope_theta = full_theta;
        } else {
            return Status::invalid_config_status("unsupported layer type");
        }
    }

    const Json* per_layer = find_field(text, "per_layer_config");
    if (per_layer == nullptr || !per_layer->is_object()) {
        return Status::invalid_config_status("missing per-layer config");
    }
    std::vector<bool> seen(config.n_layers, false);
    for (auto it = per_layer->begin(); it != per_layer->end(); ++it) {
        uint32_t index = 0;
        status = parse_layer_index(it.key(), config.n_layers, index);
        if (!status) {
            return status;
        }
        if (seen[index]) {
            return Status::invalid_config_status("duplicate per-layer config");
        }
        seen[index] = true;
        if (!it.value().is_object() || it.value().size() != 2) {
            return Status::invalid_config_status("per-layer config must set head dim and kv heads");
        }
        uint32_t head_dim = 0;
        uint32_t kv_heads = 0;
        status = require_u32(it.value(), "head_dim", head_dim);
        if (!status) {
            return status;
        }
        status = require_u32(it.value(), "num_key_value_heads", kv_heads);
        if (!status) {
            return status;
        }
        config.layers[index].head_dim = head_dim;
        config.layers[index].n_kv_heads = kv_heads;
    }

    return config.validate();
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
    if (dtype_value == nullptr || shape_value == nullptr || offsets_value == nullptr ||
        !dtype_value->is_string()) {
        return Status::invalid_argument_status("safetensors tensor entry is missing fields");
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
        if (!checked_mul(numel, item_size, expected_bytes) || out.end - out.begin != expected_bytes) {
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
    if (header_len == 0 || header_len > kMaxSafetensorsHeaderBytes ||
        header_len > header.file_size - kSafetensorsPrefixBytes) {
        return {Status::invalid_argument_status("invalid safetensors header length"), {}};
    }
    if (!checked_add(kSafetensorsPrefixBytes, header_len, header.data_base)) {
        return {Status::invalid_argument_status("safetensors data base overflow"), {}};
    }
    std::string header_json(static_cast<size_t>(header_len), '\0');
    if (!read_exact_at(file, kSafetensorsPrefixBytes, header_json.data(), header_json.size())) {
        return {Status::invalid_argument_status("failed to read safetensors header"), {}};
    }
    Result<Json> parsed_json = parse_json_text(header_json);
    if (!parsed_json.status || !parsed_json.value.is_object()) {
        return {Status::invalid_argument_status("safetensors header must be an object"), {}};
    }
    for (auto it = parsed_json.value.begin(); it != parsed_json.value.end(); ++it) {
        if (it.key() == "__metadata__") {
            continue;
        }
        SafeTensorEntry entry;
        Status status = parse_safe_tensor_entry(it.value(), header.data_base, header.file_size, entry);
        if (!status) {
            return {status, {}};
        }
        header.tensors.emplace(it.key(), std::move(entry));
    }
    return {Status::success(), std::move(header)};
}

bool shape_matches(const TensorView& view, const SafeTensorEntry& entry) {
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

Status read_tensor_as_f32(
    std::ifstream& file,
    const SafeTensorsHeader& header,
    const SafeTensorEntry& entry,
    std::vector<float>& out) {
    if (entry.dtype != SafeDType::f32 && entry.dtype != SafeDType::bf16) {
        return Status::unimplemented_status("unsupported checkpoint dtype");
    }
    uint64_t numel = 0;
    Status status = shape_numel(entry.shape, numel);
    if (!status) {
        return status;
    }
    if (numel > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return Status::invalid_argument_status("tensor is too large to load");
    }
    uint64_t file_begin = 0;
    if (!checked_add(header.data_base, entry.begin, file_begin)) {
        return Status::invalid_argument_status("tensor file offset overflow");
    }
    const size_t count = static_cast<size_t>(numel);
    out.assign(count, 0.0f);
    if (entry.dtype == SafeDType::f32) {
        if (!read_exact_at(file, file_begin, out.data(), count * sizeof(float))) {
            return Status::invalid_argument_status("failed to read tensor data");
        }
        return Status::success();
    }

    std::vector<uint16_t> bf16(count);
    if (!read_exact_at(file, file_begin, bf16.data(), count * sizeof(uint16_t))) {
        return Status::invalid_argument_status("failed to read tensor data");
    }
    for (size_t i = 0; i < count; ++i) {
        const uint32_t bits = static_cast<uint32_t>(bf16[i]) << 16;
        std::memcpy(&out[i], &bits, sizeof(float));
    }
    return Status::success();
}

void add_expected(std::vector<ExpectedTensor>& expected, const std::string& key, TensorView& view) {
    expected.push_back({key, &view});
}

std::string layer_key(uint32_t layer, const char* name) {
    return "language_model.layers." + std::to_string(layer) + "." + name;
}

}  // namespace

Result<EmbeddingGemma2Config> load_embeddinggemma2_config_json(const char* path) {
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

    const Json* model_type = find_field(parsed.value, "model_type");
    if (model_type == nullptr || !model_type->is_string()) {
        return {Status::invalid_config_status("missing model_type"), {}};
    }
    const std::string model_name = model_type->get<std::string>();
    const Json* text_config = &parsed.value;
    if (model_name == "embedding_gemma2") {
        text_config = find_field(parsed.value, "text_config");
        if (text_config == nullptr) {
            return {Status::invalid_config_status("missing text_config"), {}};
        }
    } else if (model_name != "embedding_gemma2_text") {
        return {Status::invalid_config_status("unsupported model_type"), {}};
    }

    EmbeddingGemma2Config config;
    Status status = parse_text_config(*text_config, config);
    if (!status) {
        return {status, {}};
    }
    return {Status::success(), std::move(config)};
}

Status load_embeddinggemma2_safetensors(EmbeddingGemma2Engine& engine, const char* path) {
    // After quantize, matrix views are packed int8 with a per-row scale and zero.
    // Copying f32 bytes into those views would leave quant mode set and the
    // scales stale. Reload is refused until the engine is still float32.
    if (engine.quant_ != EmbeddingGemma2Quant::f32) {
        return Status::invalid_argument_status("cannot reload weights into a quantized embedder");
    }
    engine.weights_loaded_ = false;
    if (engine.backend_ == nullptr || engine.model_.layers.size() != engine.config_.n_layers) {
        return Status::invalid_argument_status("embedder is not initialized");
    }

    Result<SafeTensorsHeader> header = parse_safetensors_header(path);
    if (!header.status) {
        return header.status;
    }

    std::vector<ExpectedTensor> expected;
    expected.reserve(5 + engine.config_.n_layers * 17);
    add_expected(expected, "language_model.embed_tokens.weight", engine.model_.embed_tokens.weight);
    add_expected(expected, "language_model.norm.weight", engine.model_.final_norm);
    add_expected(expected, "language_model.embedding_projection.weight", engine.model_.embedding_projection.weight);
    add_expected(expected, "language_model.ple.per_layer_model_projection.weight", engine.model_.ple_projection.weight);
    add_expected(expected, "language_model.ple.per_layer_projection_norm.weight", engine.model_.ple_norm);
    for (uint32_t i = 0; i < engine.config_.n_layers; ++i) {
        EmbeddingGemma2Engine::LayerWeights& layer = engine.model_.layers[i];
        add_expected(expected, layer_key(i, "input_layernorm.weight"), layer.input_layernorm);
        add_expected(expected, layer_key(i, "post_attention_layernorm.weight"), layer.post_attention_layernorm);
        add_expected(expected, layer_key(i, "pre_feedforward_layernorm.weight"), layer.pre_feedforward_layernorm);
        add_expected(expected, layer_key(i, "post_feedforward_layernorm.weight"), layer.post_feedforward_layernorm);
        add_expected(expected, layer_key(i, "self_attn.q_proj.weight"), layer.q_proj.weight);
        add_expected(expected, layer_key(i, "self_attn.k_proj.weight"), layer.k_proj.weight);
        add_expected(expected, layer_key(i, "self_attn.v_proj.weight"), layer.v_proj.weight);
        add_expected(expected, layer_key(i, "self_attn.o_proj.weight"), layer.o_proj.weight);
        add_expected(expected, layer_key(i, "self_attn.q_norm.weight"), layer.q_norm);
        add_expected(expected, layer_key(i, "self_attn.k_norm.weight"), layer.k_norm);
        add_expected(expected, layer_key(i, "mlp.gate_proj.weight"), layer.gate_proj.weight);
        add_expected(expected, layer_key(i, "mlp.up_proj.weight"), layer.up_proj.weight);
        add_expected(expected, layer_key(i, "mlp.down_proj.weight"), layer.down_proj.weight);
        add_expected(expected, layer_key(i, "ple_block.per_layer_input_gate.weight"), layer.ple_gate.weight);
        add_expected(expected, layer_key(i, "ple_block.per_layer_projection.weight"), layer.ple_proj.weight);
        add_expected(expected, layer_key(i, "ple_block.post_per_layer_input_norm.weight"), layer.ple_norm);
        add_expected(expected, layer_key(i, "layer_scalar"), layer.layer_scalar);
    }

    std::unordered_set<std::string> expected_keys;
    expected_keys.reserve(expected.size());
    for (const ExpectedTensor& item : expected) {
        expected_keys.insert(item.key);
    }
    for (const auto& item : header.value.tensors) {
        if (item.first.rfind("language_model.", 0) != 0) {
            continue;
        }
        if (!expected_keys.contains(item.first)) {
            return Status::invalid_argument_status("unexpected language_model tensor");
        }
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return Status::invalid_argument_status("failed to open safetensors file");
    }
    for (const ExpectedTensor& item : expected) {
        const auto found = header.value.tensors.find(item.key);
        if (found == header.value.tensors.end()) {
            return Status::invalid_argument_status("checkpoint is missing a text tensor");
        }
        if (item.view == nullptr || !shape_matches(*item.view, found->second)) {
            return Status::invalid_argument_status("checkpoint tensor shape mismatch");
        }
        std::vector<float> values;
        Status status = read_tensor_as_f32(file, header.value, found->second, values);
        if (!status) {
            return status;
        }
        status = engine.backend_->copy_from_host(
            *item.view,
            values.data(),
            item.view->logical_nbytes());
        if (!status) {
            return status;
        }
    }

    engine.weights_loaded_ = true;
    return Status::success();
}

Result<EmbeddingGemma2Engine> load_embeddinggemma2_from_hf_dir(
    Backend& backend,
    const char* model_dir,
    uint32_t max_seq_len) {
    std::string config_path;
    std::string weight_path;
    if (!path_join(model_dir, "config.json", config_path) ||
        !path_join(model_dir, "model.safetensors", weight_path)) {
        return {Status::invalid_argument_status("model directory path is empty"), {}};
    }

    Result<EmbeddingGemma2Config> config = load_embeddinggemma2_config_json(config_path.c_str());
    if (!config.status) {
        return {config.status, {}};
    }
    Result<EmbeddingGemma2Engine> engine = EmbeddingGemma2Engine::create(
        backend,
        config.value,
        max_seq_len);
    if (!engine.status) {
        return {engine.status, {}};
    }
    Status status = load_embeddinggemma2_safetensors(engine.value, weight_path.c_str());
    if (!status) {
        return {status, {}};
    }
    return {Status::success(), std::move(engine.value)};
}

}  // namespace tinyinfer
