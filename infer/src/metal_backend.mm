#include "tinyinfer/backend.h"

namespace tinyinfer {
namespace {

class MetalBackend final : public Backend {
public:
    Device device() const override {
        return {DeviceType::metal, 0};
    }

    Status alloc_arena(
        MemoryArena& arena,
        size_t bytes,
        MemoryKind kind) override {
        (void)arena;
        (void)bytes;
        (void)kind;
        return Status::unimplemented_status("Metal arena allocation is not implemented yet");
    }

    Status copy_from_host(
        const TensorView& dst,
        const void* src,
        size_t bytes) override {
        (void)dst;
        (void)src;
        (void)bytes;
        return Status::unimplemented_status("Metal host-to-device copy is not implemented yet");
    }

    Status copy_to_host(
        void* dst,
        const TensorView& src,
        size_t bytes) override {
        (void)dst;
        (void)src;
        (void)bytes;
        return Status::unimplemented_status("Metal device-to-host copy is not implemented yet");
    }

    Status matmul_out(
        const TensorView& out,
        const TensorView& x,
        const TensorView& w) override {
        (void)out;
        (void)x;
        (void)w;
        return Status::unimplemented_status("Metal matmul is not implemented yet");
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
        return Status::unimplemented_status("Metal RMSNorm is not implemented yet");
    }

    Status rope_inplace(
        const TensorView& q,
        const TensorView& k,
        uint32_t start_pos) override {
        (void)q;
        (void)k;
        (void)start_pos;
        return Status::unimplemented_status("Metal RoPE is not implemented yet");
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
        return Status::unimplemented_status("Metal attention is not implemented yet");
    }

    Status swiglu_out(
        const TensorView& out,
        const TensorView& gate,
        const TensorView& up) override {
        (void)out;
        (void)gate;
        (void)up;
        return Status::unimplemented_status("Metal SwiGLU is not implemented yet");
    }

    Status argmax(
        uint32_t& out_token,
        const TensorView& logits) override {
        (void)out_token;
        (void)logits;
        return Status::unimplemented_status("Metal argmax is not implemented yet");
    }

    Status synchronize() override {
        return Status::unimplemented_status("Metal synchronize is not implemented yet");
    }

protected:
    Status release_arena(MemoryArena& arena) override {
        (void)arena;
        return Status::success();
    }
};

}  // namespace

Result<std::unique_ptr<Backend>> create_metal_backend() {
    return {Status::success(), std::make_unique<MetalBackend>()};
}

}  // namespace tinyinfer
