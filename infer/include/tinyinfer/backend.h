#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include "tinyinfer/core.h"

namespace tinyinfer {

class Backend {
public:
    virtual ~Backend() = default;

    virtual Device device() const = 0;

    virtual Status alloc_arena(
        MemoryArena& arena,
        size_t bytes,
        MemoryKind kind) = 0;

    virtual Status copy_from_host(
        const TensorView& dst,
        const void* src,
        size_t bytes) = 0;
    virtual Status copy_to_host(
        void* dst,
        const TensorView& src,
        size_t bytes) = 0;

    virtual void matmul_out(
        const TensorView& out,
        const TensorView& x,
        const TensorView& w) = 0;
    virtual void embedding_out(
        const TensorView& out,
        const TensorView& table,
        std::span<const uint32_t> token_ids) = 0;
    virtual void add_inplace(
        const TensorView& dst,
        const TensorView& src) = 0;
    virtual void rms_norm_out(
        const TensorView& out,
        const TensorView& x,
        const TensorView& weight,
        float eps) = 0;
    virtual void rope_inplace(
        const TensorView& q,
        const TensorView& k,
        uint32_t start_pos,
        float theta) = 0;
    virtual void attention_out(
        const TensorView& out,
        const TensorView& q,
        const TensorView& k,
        const TensorView& v,
        const TensorView& k_cache,
        const TensorView& v_cache,
        uint32_t start_pos,
        uint32_t kv_len) = 0;
    virtual void swiglu_out(
        const TensorView& out,
        const TensorView& gate,
        const TensorView& up) = 0;
    virtual void argmax(
        uint32_t& out_token,
        const TensorView& logits) = 0;

    virtual Status synchronize() = 0;

protected:
    friend class MemoryArena;

    void bind_arena(
        MemoryArena& arena,
        void* native_handle,
        size_t bytes,
        MemoryKind kind);
    bool owns_arena(const MemoryArena& arena) const;
    void* arena_handle(const MemoryArena& arena) const;

    virtual void release_arena(MemoryArena& arena) noexcept = 0;
};

Result<std::unique_ptr<Backend>> create_cpu_backend();
Result<std::unique_ptr<Backend>> create_metal_backend();

}  // namespace tinyinfer
