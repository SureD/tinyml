#include "tinyinfer/backend.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>

namespace tinyinfer {
namespace {

constexpr size_t kArenaAlignment = 64;

class CPUBackend final : public Backend {
public:
    Device device() const override {
        return {DeviceType::cpu, 0};
    }

    Status alloc_arena(
        MemoryArena& arena,
        size_t bytes,
        MemoryKind kind) override {
        if (bytes == 0) {
            return Status::invalid_argument_status("CPU arena size must be non-zero");
        }

        void* handle = ::operator new(
            bytes,
            std::align_val_t(kArenaAlignment),
            std::nothrow);
        if (handle == nullptr) {
            return Status::backend_error_status("CPU arena allocation failed");
        }

        bind_arena(arena, handle, bytes, kind);
        return Status::success();
    }

    Status copy_from_host(
        const TensorView& dst,
        const void* src,
        size_t bytes) override {
        Status status = validate_copy_view(dst, bytes);
        if (!status) {
            return status;
        }
        if (bytes == 0) {
            return Status::success();
        }
        if (src == nullptr) {
            return Status::invalid_argument_status("source host pointer is null");
        }

        std::memcpy(data(dst), src, bytes);
        return Status::success();
    }

    Status copy_to_host(
        void* dst,
        const TensorView& src,
        size_t bytes) override {
        Status status = validate_copy_view(src, bytes);
        if (!status) {
            return status;
        }
        if (bytes == 0) {
            return Status::success();
        }
        if (dst == nullptr) {
            return Status::invalid_argument_status("destination host pointer is null");
        }

        std::memcpy(dst, data(src), bytes);
        return Status::success();
    }

    Status matmul_out(
        const TensorView& out,
        const TensorView& x,
        const TensorView& w) override {
        (void)out;
        (void)x;
        (void)w;
        return Status::unimplemented_status("CPU matmul is not implemented yet");
    }

    Status rms_norm_out(
        const TensorView& out,
        const TensorView& x,
        const TensorView& weight,
        float eps) override {
        (void)out;
        (void)x;
        (void)weight;
        (void)eps;
        return Status::unimplemented_status("CPU RMSNorm is not implemented yet");
    }

    Status rope_inplace(
        const TensorView& q,
        const TensorView& k,
        uint32_t start_pos) override {
        (void)q;
        (void)k;
        (void)start_pos;
        return Status::unimplemented_status("CPU RoPE is not implemented yet");
    }

    Status attention_out(
        const TensorView& out,
        const TensorView& q,
        const TensorView& k_cache,
        const TensorView& v_cache,
        uint32_t kv_len) override {
        (void)out;
        (void)q;
        (void)k_cache;
        (void)v_cache;
        (void)kv_len;
        return Status::unimplemented_status("CPU attention is not implemented yet");
    }

    Status swiglu_out(
        const TensorView& out,
        const TensorView& gate,
        const TensorView& up) override {
        (void)out;
        (void)gate;
        (void)up;
        return Status::unimplemented_status("CPU SwiGLU is not implemented yet");
    }

    Status argmax(
        uint32_t& out_token,
        const TensorView& logits) override {
        (void)out_token;
        (void)logits;
        return Status::unimplemented_status("CPU argmax is not implemented yet");
    }

    Status synchronize() override {
        return Status::success();
    }

protected:
    Status release_arena(MemoryArena& arena) override {
        void* handle = arena_handle(arena);
        if (handle != nullptr) {
            ::operator delete(handle, std::align_val_t(kArenaAlignment));
        }
        return Status::success();
    }

private:
    Status validate_copy_view(const TensorView& view, size_t bytes) const {
        if (!view.defined()) {
            return Status::invalid_argument_status("tensor view is not defined");
        }
        if (!owns_arena(*view.arena)) {
            return Status::invalid_argument_status("tensor view belongs to a different backend");
        }
        if (view.device().type != DeviceType::cpu) {
            return Status::invalid_argument_status("tensor view is not on CPU");
        }
        if (!view.contiguous()) {
            return Status::invalid_argument_status("copy requires a contiguous tensor view");
        }
        if (bytes > view.logical_nbytes()) {
            return Status::invalid_argument_status("copy exceeds tensor logical byte size");
        }
        return Status::success();
    }

    uint8_t* data(const TensorView& view) {
        return static_cast<uint8_t*>(arena_handle(*view.arena)) + view.byte_offset;
    }

    const uint8_t* data(const TensorView& view) const {
        return static_cast<const uint8_t*>(arena_handle(*view.arena)) + view.byte_offset;
    }
};

}  // namespace

Result<std::unique_ptr<Backend>> create_cpu_backend() {
    return {Status::success(), std::make_unique<CPUBackend>()};
}

}  // namespace tinyinfer
