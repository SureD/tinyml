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

- every backend operation commits and waits for its own command buffer
- matmul assigns one output element to one GPU thread and has no tiling
- attention recomputes scores during max, denominator, and value accumulation
- all tensors use f32
- the benchmark was built with `CMAKE_BUILD_TYPE=Debug`

Future measurements should use a Release build, at least five repetitions per
case, median and percentile reporting, and the same deterministic token inputs.
