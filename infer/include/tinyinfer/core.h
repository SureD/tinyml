#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace tinyinfer {

class Backend;
struct TensorView;

enum class DType {
    f32,
    f16,
    bf16,
    i32,
    u32,
};

enum class DeviceType {
    cpu,
    metal,
};

enum class MemoryKind {
    weights,
    kv_cache,
    workspace,
    host,
};

struct Status {
    enum Code {
        ok,
        invalid_argument,
        invalid_config,
        backend_error,
        unimplemented,
    };

    Code code = ok;
    const char* message = "";

    explicit operator bool() const { return code == ok; }
    bool is_ok() const { return code == ok; }

    static Status success() { return {ok, ""}; }
    static Status invalid_argument_status(const char* message) {
        return {invalid_argument, message};
    }
    static Status invalid_config_status(const char* message) {
        return {invalid_config, message};
    }
    static Status backend_error_status(const char* message) {
        return {backend_error, message};
    }
    static Status unimplemented_status(const char* message) {
        return {unimplemented, message};
    }
};

template <class T>
struct Result {
    Status status;
    T value;

    explicit operator bool() const { return static_cast<bool>(status); }
};

size_t dtype_size(DType dtype);

struct Shape {
    static constexpr uint32_t kMaxDims = 8;

    uint32_t ndim = 0;
    int64_t dims[kMaxDims] = {};

    int64_t dim(uint32_t i) const;
    int64_t numel() const;
    bool valid() const;
};

Shape make_shape(std::initializer_list<int64_t> dims);

struct Strides {
    static constexpr uint32_t kMaxDims = 8;

    uint32_t ndim = 0;
    int64_t values[kMaxDims] = {};

    int64_t stride(uint32_t i) const;
};

Strides contiguous_strides(const Shape& shape);

struct Device {
    DeviceType type = DeviceType::cpu;
    int index = 0;
};

class MemoryArena {
public:
    MemoryArena() = default;
    MemoryArena(MemoryArena&& other) noexcept;
    MemoryArena& operator=(MemoryArena&& other) noexcept;
    MemoryArena(const MemoryArena&) = delete;
    MemoryArena& operator=(const MemoryArena&) = delete;
    ~MemoryArena();

    Result<TensorView> alloc(const Shape& shape, DType dtype);

    void reset();

    bool defined() const;
    MemoryKind kind() const;
    Device device() const;
    size_t used() const;
    size_t capacity() const;

private:
    friend class Backend;

    void release() noexcept;

    Backend* backend_ = nullptr;
    void* handle_ = nullptr;
    size_t capacity_ = 0;
    size_t offset_ = 0;
    MemoryKind kind_ = MemoryKind::workspace;
    Device device_;
};

struct TensorView {
    MemoryArena* arena = nullptr;
    size_t byte_offset = 0;

    DType dtype = DType::f32;
    Shape shape;
    Strides strides;

    int64_t dim(uint32_t i) const;
    int64_t numel() const;
    size_t item_size() const;
    size_t logical_nbytes() const;
    size_t storage_span_nbytes() const;

    bool defined() const;
    bool contiguous() const;
    Device device() const;
};

}  // namespace tinyinfer
