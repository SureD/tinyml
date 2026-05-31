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

    void matmul_out(
        const TensorView& out,
        const TensorView& x,
        const TensorView& w) override {
        (void)out;
        (void)x;
        (void)w;
        panic("Metal matmul is not implemented yet", __FILE__, __LINE__);
    }

    void embedding_out(
        const TensorView& out,
        const TensorView& table,
        std::span<const uint32_t> token_ids) override {
        (void)out;
        (void)table;
        (void)token_ids;
        panic("Metal embedding lookup is not implemented yet", __FILE__, __LINE__);
    }

    void add_inplace(
        const TensorView& dst,
        const TensorView& src) override {
        (void)dst;
        (void)src;
        panic("Metal add is not implemented yet", __FILE__, __LINE__);
    }

    void rms_norm_out(
        const TensorView& out,
        const TensorView& x,
        const TensorView& weight,
        float eps) override {
        (void)out;
        (void)x;
        (void)weight;
        (void)eps;
        panic("Metal RMSNorm is not implemented yet", __FILE__, __LINE__);
    }

    void rope_inplace(
        const TensorView& q,
        const TensorView& k,
        uint32_t start_pos,
        float theta) override {
        (void)q;
        (void)k;
        (void)start_pos;
        (void)theta;
        panic("Metal RoPE is not implemented yet", __FILE__, __LINE__);
    }

    void attention_out(
        const TensorView& out,
        const TensorView& q,
        const TensorView& k,
        const TensorView& v,
        const TensorView& k_cache,
        const TensorView& v_cache,
        uint32_t start_pos,
        uint32_t kv_len) override {
        (void)out;
        (void)q;
        (void)k;
        (void)v;
        (void)k_cache;
        (void)v_cache;
        (void)start_pos;
        (void)kv_len;
        panic("Metal attention is not implemented yet", __FILE__, __LINE__);
    }

    void swiglu_out(
        const TensorView& out,
        const TensorView& gate,
        const TensorView& up) override {
        (void)out;
        (void)gate;
        (void)up;
        panic("Metal SwiGLU is not implemented yet", __FILE__, __LINE__);
    }

    void argmax(
        uint32_t& out_token,
        const TensorView& logits) override {
        (void)out_token;
        (void)logits;
        panic("Metal argmax is not implemented yet", __FILE__, __LINE__);
    }

    Status synchronize() override {
        return Status::unimplemented_status("Metal synchronize is not implemented yet");
    }

protected:
    void release_arena(MemoryArena& arena) noexcept override {
        (void)arena;
    }
};

}  // namespace

Result<std::unique_ptr<Backend>> create_metal_backend() {
    return {Status::success(), std::make_unique<MetalBackend>()};
}

}  // namespace tinyinfer
