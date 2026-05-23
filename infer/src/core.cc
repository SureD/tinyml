#include "tinyinfer/core.h"

#include <cstddef>
#include <cstdint>
#include <limits>

#include "tinyinfer/backend.h"

namespace tinyinfer {
namespace {

bool checked_add(size_t a, size_t b, size_t& out) {
    if (a > std::numeric_limits<size_t>::max() - b) {
        return false;
    }
    out = a + b;
    return true;
}

bool checked_mul(size_t a, size_t b, size_t& out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

bool checked_align_up(size_t value, size_t alignment, size_t& out) {
    if (alignment == 0) {
        out = value;
        return true;
    }
    const size_t rem = value % alignment;
    if (rem == 0) {
        out = value;
        return true;
    }
    return checked_add(value, alignment - rem, out);
}

}  // namespace

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
    return (i < ndim && i < kMaxDims) ? dims[i] : 0;
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
    if (ndim > kMaxDims) {
        return false;
    }
    int64_t total = 1;
    for (uint32_t i = 0; i < ndim; ++i) {
        if (dims[i] <= 0) {
            return false;
        }
        if (dims[i] > std::numeric_limits<int64_t>::max() / total) {
            return false;
        }
        total *= dims[i];
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
    return (i < ndim && i < kMaxDims) ? values[i] : 0;
}

Strides contiguous_strides(const Shape& shape) {
    Strides strides;
    if (!shape.valid()) {
        return strides;
    }

    strides.ndim = shape.ndim;

    int64_t stride = 1;
    for (uint32_t i = shape.ndim; i > 0; --i) {
        const uint32_t idx = i - 1;
        strides.values[idx] = stride;
        stride *= shape.dims[idx];
    }
    return strides;
}

MemoryArena::MemoryArena(MemoryArena&& other) noexcept {
    backend_ = other.backend_;
    handle_ = other.handle_;
    capacity_ = other.capacity_;
    offset_ = other.offset_;
    kind_ = other.kind_;
    device_ = other.device_;

    other.backend_ = nullptr;
    other.handle_ = nullptr;
    other.capacity_ = 0;
    other.offset_ = 0;
}

MemoryArena& MemoryArena::operator=(MemoryArena&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    release();

    backend_ = other.backend_;
    handle_ = other.handle_;
    capacity_ = other.capacity_;
    offset_ = other.offset_;
    kind_ = other.kind_;
    device_ = other.device_;

    other.backend_ = nullptr;
    other.handle_ = nullptr;
    other.capacity_ = 0;
    other.offset_ = 0;

    return *this;
}

MemoryArena::~MemoryArena() {
    release();
}

Result<TensorView> MemoryArena::alloc(
    const Shape& shape,
    DType dtype,
    size_t alignment) {
    if (!defined()) {
        return {Status::invalid_argument_status("arena is not allocated"), {}};
    }
    if (!shape.valid()) {
        return {Status::invalid_argument_status("invalid tensor shape"), {}};
    }
    const size_t item_bytes = dtype_size(dtype);
    if (item_bytes == 0) {
        return {Status::invalid_argument_status("invalid dtype"), {}};
    }

    const int64_t elements = shape.numel();
    if (elements <= 0) {
        return {Status::invalid_argument_status("invalid tensor size"), {}};
    }

    size_t start = 0;
    if (!checked_align_up(offset_, alignment, start)) {
        return {Status::invalid_argument_status("arena offset overflow"), {}};
    }

    size_t bytes = 0;
    if (!checked_mul(static_cast<size_t>(elements), item_bytes, bytes)) {
        return {Status::invalid_argument_status("tensor byte size overflow"), {}};
    }
    if (bytes > capacity_ || start > capacity_ - bytes) {
        return {Status::invalid_argument_status("arena capacity exceeded"), {}};
    }

    TensorView view;
    view.arena = this;
    view.byte_offset = start;
    view.dtype = dtype;
    view.shape = shape;
    view.strides = contiguous_strides(shape);

    offset_ = start + bytes;
    return {Status::success(), view};
}

void MemoryArena::reset() {
    offset_ = 0;
}

bool MemoryArena::defined() const {
    return backend_ != nullptr && handle_ != nullptr && capacity_ != 0;
}

MemoryKind MemoryArena::kind() const {
    return kind_;
}

Device MemoryArena::device() const {
    return device_;
}

size_t MemoryArena::used() const {
    return offset_;
}

size_t MemoryArena::capacity() const {
    return capacity_;
}

void MemoryArena::release() noexcept {
    if (backend_ != nullptr) {
        (void)backend_->release_arena(*this);
    }

    backend_ = nullptr;
    handle_ = nullptr;
    capacity_ = 0;
    offset_ = 0;
}

int64_t TensorView::dim(uint32_t i) const {
    return shape.dim(i);
}

int64_t TensorView::numel() const {
    return shape.numel();
}

size_t TensorView::item_size() const {
    return dtype_size(dtype);
}

size_t TensorView::logical_nbytes() const {
    const int64_t elements = shape.numel();
    if (elements <= 0) {
        return 0;
    }
    size_t bytes = 0;
    return checked_mul(static_cast<size_t>(elements), item_size(), bytes) ? bytes : 0;
}

size_t TensorView::storage_span_nbytes() const {
    if (!shape.valid() || shape.ndim != strides.ndim) {
        return 0;
    }

    size_t max_elem_offset = 0;
    for (uint32_t i = 0; i < shape.ndim; ++i) {
        if (strides.values[i] < 0) {
            return 0;
        }
        size_t dim_offset = 0;
        if (!checked_mul(
                static_cast<size_t>(shape.dims[i] - 1),
                static_cast<size_t>(strides.values[i]),
                dim_offset)) {
            return 0;
        }
        if (!checked_add(max_elem_offset, dim_offset, max_elem_offset)) {
            return 0;
        }
    }

    size_t element_count = 0;
    if (!checked_add(max_elem_offset, 1, element_count)) {
        return 0;
    }

    size_t bytes = 0;
    return checked_mul(element_count, item_size(), bytes) ? bytes : 0;
}

bool TensorView::defined() const {
    if (arena == nullptr || !arena->defined()) {
        return false;
    }

    const size_t span = storage_span_nbytes();
    return span != 0 && byte_offset <= arena->capacity() &&
           span <= arena->capacity() - byte_offset;
}

bool TensorView::contiguous() const {
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

Device TensorView::device() const {
    return arena == nullptr ? Device{} : arena->device();
}

void Backend::bind_arena(
    MemoryArena& arena,
    void* native_handle,
    size_t bytes,
    MemoryKind kind) {
    arena.release();
    arena.backend_ = this;
    arena.handle_ = native_handle;
    arena.capacity_ = bytes;
    arena.offset_ = 0;
    arena.kind_ = kind;
    arena.device_ = device();
}

void* Backend::arena_handle(const MemoryArena& arena) const {
    return arena.handle_;
}

}  // namespace tinyinfer
