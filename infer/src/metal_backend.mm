#include "tinyinfer/backend.h"

namespace tinyinfer {
namespace {

class MetalBackend final : public Backend {
public:
    Device device() const override {
        return {DeviceType::metal, 0};
    }

    Result<Stream> new_stream() override {
        return {Status::success(), make_stream()};
    }

    Result<Tensor> empty(const Shape& shape, DType dtype) override {
        (void)shape;
        (void)dtype;
        return {{Status::unimplemented, "Metal allocation is not implemented yet"}, {}};
    }

    Status copy_from_host(
        Tensor& dst,
        const void* src,
        size_t bytes,
        Stream& stream) override {
        (void)dst;
        (void)src;
        (void)bytes;
        (void)stream;
        return Status::unimplemented_status("Metal host-to-device copy is not implemented yet");
    }

    Status copy_to_host(
        void* dst,
        const Tensor& src,
        size_t bytes,
        Stream& stream) override {
        (void)dst;
        (void)src;
        (void)bytes;
        (void)stream;
        return Status::unimplemented_status("Metal device-to-host copy is not implemented yet");
    }

    Status matmul_out(
        Tensor& out,
        const Tensor& x,
        const Tensor& w,
        Stream& stream) override {
        (void)out;
        (void)x;
        (void)w;
        (void)stream;
        return Status::unimplemented_status("Metal matmul is not implemented yet");
    }

    Status rms_norm_out(
        Tensor& out,
        const Tensor& x,
        const Tensor& weight,
        float eps,
        Stream& stream) override {
        (void)out;
        (void)x;
        (void)weight;
        (void)eps;
        (void)stream;
        return Status::unimplemented_status("Metal RMSNorm is not implemented yet");
    }

    Status rope_inplace(
        Tensor& q,
        Tensor& k,
        uint32_t start_pos,
        Stream& stream) override {
        (void)q;
        (void)k;
        (void)start_pos;
        (void)stream;
        return Status::unimplemented_status("Metal RoPE is not implemented yet");
    }

    Status attention_out(
        Tensor& out,
        const Tensor& q,
        const Tensor& k_cache,
        const Tensor& v_cache,
        uint32_t kv_len,
        Stream& stream) override {
        (void)out;
        (void)q;
        (void)k_cache;
        (void)v_cache;
        (void)kv_len;
        (void)stream;
        return Status::unimplemented_status("Metal attention is not implemented yet");
    }

    Status swiglu_out(
        Tensor& out,
        const Tensor& gate,
        const Tensor& up,
        Stream& stream) override {
        (void)out;
        (void)gate;
        (void)up;
        (void)stream;
        return Status::unimplemented_status("Metal SwiGLU is not implemented yet");
    }

    Status argmax(
        uint32_t& out_token,
        const Tensor& logits,
        Stream& stream) override {
        (void)out_token;
        (void)logits;
        (void)stream;
        return Status::unimplemented_status("Metal argmax is not implemented yet");
    }

    Status synchronize(Stream& stream) override {
        (void)stream;
        return Status::unimplemented_status("Metal stream synchronize is not implemented yet");
    }

protected:
    Status release_storage(Storage& storage) override {
        (void)storage;
        return Status::success();
    }
};

}  // namespace

Result<std::unique_ptr<Backend>> create_metal_backend() {
    return {Status::success(), std::make_unique<MetalBackend>()};
}

}  // namespace tinyinfer
