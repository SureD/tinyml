# Metal Backend Validation

Date: 2026-06-06

## Environment

- Device: Apple M5
- OS: macOS 26.5.1 (25F80)
- Architecture: arm64
- Compiler: Apple clang 21.0.0
- CMake build type: Debug
- Base commit: `3e6b74b`, with uncommitted Metal backend changes
- Model: `models/TinyLlama-1.1B-Chat-v1.0`
- Model dtype in the runtime: f32

## Correctness

### Operator and Small-Model Tests

Command:

```sh
ctest --test-dir build --output-on-failure
```

Result:

```text
100% tests passed, 0 tests failed out of 1
```

The Metal tests execute real MSL kernels and cover:

- host-to-Metal and Metal-to-host copies
- matmul
- embedding lookup
- residual add
- RMSNorm
- RoPE
- KV cache writes
- causal grouped-query attention
- SwiGLU
- argmax
- CPU/Metal token parity on a deterministic one-layer LLaMA fixture

### TinyLlama Token Parity

Input:

```text
prompt_tokens: 1 15043 29892 590
max_new_tokens: 2
```

Commands:

```sh
./build/run_token_ids_smoke --backend cpu \
  models/TinyLlama-1.1B-Chat-v1.0 16 2 1 15043 29892 590

./build/run_token_ids_smoke --backend metal \
  models/TinyLlama-1.1B-Chat-v1.0 16 2 1 15043 29892 590
```

Results:

| Backend | Generated tokens | Match |
| --- | --- | --- |
| CPU | `1024 338` | yes |
| Metal | `1024 338` | yes |

This verifies one real TinyLlama prefill followed by one decode step. It is not
an exhaustive numerical parity test over long contexts or many prompts.

## Performance

The benchmark performs an untimed warmup before measuring. The following
numbers are one run of the fixed `p16_d8` case, not a multi-run statistical
result.

Command:

```sh
python3 scripts/bench_infer_cases.py --backend cpu --case p16_d8 --no-build
python3 scripts/bench_infer_cases.py --backend metal --case p16_d8 --no-build
```

Case:

- prompt length: 16
- maximum sequence length: 128
- generated tokens: 8
- deterministic seed: `0xc0ffee10`

| Backend | Prefill | Decode total | Decode/token | Tokens/s |
| --- | ---: | ---: | ---: | ---: |
| CPU | 18064.133 ms | 9845.423 ms | 1230.678 ms | 0.813 |
| Metal | 628.385 ms | 1258.979 ms | 157.372 ms | 6.354 |

Observed speedup:

- prefill: 28.74x
- decode latency: 7.82x
- decode throughput: 7.82x

## Interpretation

This is a correctness-first Metal implementation, not an optimized one:

- one inference step batches its dispatches into one compute encoder and waits
  when the final token argmax needs to return to the CPU
- matmul has dedicated decode and small-M paths, while `M > 16` still uses a
  generic one-output-element-per-thread fallback
- attention recomputes scores during max, denominator, and value accumulation
- all tensors use f32
- the benchmark was built with `CMAKE_BUILD_TYPE=Debug`

Future measurements should use a Release build, at least five repetitions per
case, median and percentile reporting, and the same deterministic token inputs.

## Profiling

### Lightweight GPU Timing

Set `TINYINFER_METAL_PROFILE=1` to label Metal command buffers and encoders,
insert kernel debug signposts, and print the completed command buffer's GPU
execution span:

```sh
TINYINFER_METAL_PROFILE=1 \
./build/run_token_ids_smoke --backend metal \
  models/TinyLlama-1.1B-Chat-v1.0 16 2 1 15043 29892 590
```

Example output on stderr:

```text
metal_profile,label=tinyinfer.prefill,gpu_ms=...
metal_profile,label=tinyinfer.decode,gpu_ms=...
```

This timing uses `MTLCommandBuffer.GPUStartTime` and `GPUEndTime`, so it
measures the GPU execution span after the command buffer completes. It does
not include CPU command encoding, queueing before the GPU starts, or model
loading. The regular benchmark remains the source of end-to-end prefill and
decode latency.

Profiling is disabled by default. The enabled path adds debug labels and
signposts, so do not use its numbers as the final benchmark result.

### Instruments

Install full Xcode and select its developer directory:

```sh
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
xctrace version
```

Build an optimized binary with symbols:

```sh
cmake -S . -B build-profile -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-profile -j
```

In Instruments:

1. Open the Game Performance template, or add Metal System Trace, GPU, Metal
   Application, Time Profiler, and Points of Interest instruments.
2. Select `build-profile/run_token_ids_smoke` as the target and pass the same
   arguments used above. Set `TINYINFER_METAL_PROFILE=1` in the target
   environment to enable the labels and debug signposts.
3. Enable a performance-limiter or utilization counter set in Recording
   Options.
4. Record one warm run and inspect the `tinyinfer.prefill`,
   `tinyinfer.decode`, and `tinyinfer.compute` labels.

Use the trace to answer system-level questions:

- Is the GPU busy continuously, or waiting for CPU submission?
- Is decode dominated by GPU read bandwidth?
- Does occupancy fall because dispatches are too small or shaders use too many
  registers/threadgroup resources?
- Do command buffer boundaries or CPU waits serialize work?
- Does an optimization reduce GPU time, or only CPU encoding time?

### Xcode GPU Capture

Use Xcode's Metal GPU Capture for shader-level analysis. The Performance
timeline shows individual compute shaders, their duration, occupancy, limiter
counters, and read/write bandwidth. The backend's debug signposts use the MSL
kernel names, including `matvec_f32`, `attention_f32`, and
`small_m_gemm_tiled_f32`, so captured dispatches are easy to map back to
`infer/src/metal_backend.mm`.

For each optimization, compare one representative prefill and one decode
capture:

- Prefill: inspect GEMM occupancy, ALU utilization, and bandwidth.
- Decode: inspect matvec read bandwidth and the number of small dispatches.
- Attention: increase context length and inspect how shader duration and
  bandwidth scale.
- Fusion: verify that dispatch count and intermediate memory traffic decrease.

Avoid inserting per-dispatch counter barriers in the normal path. They
serialize GPU work and can materially change the workload being measured.

Apple references:

- [Analyzing the performance of your Metal app](https://developer.apple.com/documentation/xcode/analyzing-the-performance-of-your-metal-app)
- [Analyzing Apple GPU performance using a visual timeline](https://developer.apple.com/documentation/xcode/analyzing-apple-gpu-performance-using-a-visual-timeline)
- [Naming resources and commands](https://developer.apple.com/documentation/xcode/naming-resources-and-commands)
- [Capturing a Metal workload in Xcode](https://developer.apple.com/documentation/xcode/capturing-a-metal-workload-in-xcode)
