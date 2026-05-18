#include "tinyinfer/core.h"

#include <algorithm>
#include <limits>

#include "tinyinfer/backend.h"

namespace tinyinfer {

size_t dtype_size(DType dtype) {
    switch (dtype) {
    case DType::f32:
    case DType::i32:
    case DType::u32:
        return 4;
    case DType::f16:
    case DType::bf16:
        return 2;
    }
    return 0;
}

int64_t Shape::dim(uint32_t i) const {
    return i < ndim ? dims[i] : 0;
}

int64_t Shape::numel() const {
    if (!valid()) {
        return 0;
    }

    int64_t total = 1;
    for (uint32_t i = 0; i < ndim; ++i) {
        if (dims[i] > std::numeric_limits<int64_t>::max() / total) {
            return 0;
        }
        total *= dims[i];
    }
    return total;
}

bool Shape::valid() const {
    if (ndim == 0 || ndim > kMaxDims) {
        return false;
    }
    for (uint32_t i = 0; i < ndim; ++i) {
        if (dims[i] <= 0) {
            return false;
        }
    }
    return true;
}

Shape make_shape(std::initializer_list<int64_t> dims) {
    Shape shape;
    shape.ndim = static_cast<uint32_t>(dims.size());

    uint32_t i = 0;
    for (int64_t dim : dims) {
        if (i < Shape::kMaxDims) {
            shape.dims[i] = dim;
        }
        ++i;
    }
    return shape;
}

int64_t Strides::stride(uint32_t i) const {
    return i < ndim ? values[i] : 0;
}

Strides contiguous_strides(const Shape& shape) {
    Strides strides;
    strides.ndim = shape.ndim;

    int64_t stride = 1;
    const uint32_t ndim = std::min<uint32_t>(shape.ndim, Shape::kMaxDims);
    for (uint32_t i = ndim; i > 0; --i) {
        const uint32_t idx = i - 1;
        strides.values[idx] = stride;
        stride *= std::max<int64_t>(shape.dims[idx], 1);
    }
    return strides;
}

Storage::Storage(Storage&& other) noexcept {
    backend_ = other.backend_;
    handle_ = other.handle_;
    nbytes_ = other.nbytes_;

    other.backend_ = nullptr;
    other.handle_ = nullptr;
    other.nbytes_ = 0;
}

Storage& Storage::operator=(Storage&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    release();

    backend_ = other.backend_;
    handle_ = other.handle_;
    nbytes_ = other.nbytes_;

    other.backend_ = nullptr;
    other.handle_ = nullptr;
    other.nbytes_ = 0;

    return *this;
}

Storage::~Storage() {
    release();
}

void* Storage::native_handle() const {
    return handle_;
}

size_t Storage::nbytes() const {
    return nbytes_;
}

void Storage::release() noexcept {
    if (backend_ != nullptr) {
        (void)backend_->release_storage(*this);
    }

    backend_ = nullptr;
    handle_ = nullptr;
    nbytes_ = 0;
}

int64_t Tensor::dim(uint32_t i) const {
    return shape.dim(i);
}

size_t Tensor::nbytes() const {
    if (storage.nbytes() != 0) {
        return storage.nbytes();
    }

    const int64_t elements = shape.numel();
    if (elements <= 0) {
        return 0;
    }
    return static_cast<size_t>(elements) * dtype_size(dtype);
}

bool Tensor::contiguous() const {
    if (shape.ndim != strides.ndim || !shape.valid()) {
        return false;
    }

    const Strides expected = contiguous_strides(shape);
    for (uint32_t i = 0; i < shape.ndim; ++i) {
        if (strides.values[i] != expected.values[i]) {
            return false;
        }
    }
    return true;
}

Device Stream::device() const {
    return device_;
}

void* Stream::native_handle() const {
    return native_handle_;
}

Stream Backend::make_stream(void* native_handle) const {
    Stream stream;
    stream.device_ = device();
    stream.native_handle_ = native_handle;
    return stream;
}

Tensor Backend::make_tensor(
    const Shape& shape,
    DType dtype,
    void* native_handle,
    size_t nbytes) {
    Tensor tensor;
    tensor.dtype = dtype;
    tensor.shape = shape;
    tensor.strides = contiguous_strides(shape);
    tensor.device = device();
    tensor.storage.backend_ = this;
    tensor.storage.handle_ = native_handle;
    tensor.storage.nbytes_ = nbytes;
    return tensor;
}

}  // namespace tinyinfer
