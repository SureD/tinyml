#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "tinyinfer/core.h"

namespace tinyinfer {

class Backend {
public:
    virtual ~Backend() = default;

    virtual Device device() const = 0;
    virtual Result<Stream> new_stream() = 0;

    virtual Result<Tensor> empty(const Shape& shape, DType dtype) = 0;

    virtual Status copy_from_host(
        Tensor& dst,
        const void* src,
        size_t bytes,
        Stream& stream) = 0;
    virtual Status copy_to_host(
        void* dst,
        const Tensor& src,
        size_t bytes,
        Stream& stream) = 0;

    virtual Status matmul_out(
        Tensor& out,
        const Tensor& x,
        const Tensor& w,
        Stream& stream) = 0;
    virtual Status rms_norm_out(
        Tensor& out,
        const Tensor& x,
        const Tensor& weight,
        float eps,
        Stream& stream) = 0;
    virtual Status rope_inplace(
        Tensor& q,
        Tensor& k,
        uint32_t start_pos,
        Stream& stream) = 0;
    virtual Status attention_out(
        Tensor& out,
        const Tensor& q,
        const Tensor& k_cache,
        const Tensor& v_cache,
        uint32_t kv_len,
        Stream& stream) = 0;
    virtual Status swiglu_out(
        Tensor& out,
        const Tensor& gate,
        const Tensor& up,
        Stream& stream) = 0;
    virtual Status argmax(
        uint32_t& out_token,
        const Tensor& logits,
        Stream& stream) = 0;

    virtual Status synchronize(Stream& stream) = 0;

protected:
    friend class Storage;

    Stream make_stream(void* native_handle = nullptr) const;
    Tensor make_tensor(
        const Shape& shape,
        DType dtype,
        void* native_handle,
        size_t nbytes);

    virtual Status release_storage(Storage& storage) = 0;
};

Result<std::unique_ptr<Backend>> create_metal_backend();

}  // namespace tinyinfer
