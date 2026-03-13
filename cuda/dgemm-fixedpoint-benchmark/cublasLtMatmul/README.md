# cuBLASLt FP emulation test for DGEMM

This miniapp benchmarks `cublasLtMatmul` in one run with two compute types:

- Baseline: `CUBLAS_COMPUTE_64F`
- Emulated: `CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT`

Reference: https://docs.nvidia.com/cuda/cublas/#cublasltmatmul

## Build & Run

```sh
make
./dgemm-lt-fixedpoint-benchmark.x
```

## Results (using nvcc 13.1 on RTX 3080)

- Non-batched DGEMM with `M=N=K=1024`
```
❯ ./dgemm-lt-fixedpoint-benchmark.x --m 1024 --n 1024 --k 1024 --warmup 10 --iters 50
DGEMM fixed-point emulation benchmark via cublasLtMatmul, column-major
M,N,K = 1024,1024,1024 | warmup=10 iters=50
batched=0 batch_count=1
mantissa_control=fixed max_mantissa_bits=55 mantissa_offset=0

GPU: NVIDIA GeForce RTX 3080 (cc 8.6)
cuBLAS version: 130201
Env CUBLAS_EMULATE_DOUBLE_PRECISION = (unset)
Env CUBLAS_EMULATION_STRATEGY = (unset)
Workspace bytes = 152578048 (145.51 MiB)

[algo] mode=lt_64f
 id=22 tile=13 splitK=1 reduction=0 swizzle=0

[algo] mode=lt_64f_emulated_fixedpoint
 id=65 tile=0 splitK=1 reduction=0 swizzle=0

[run] mode=lt_64f
 avg_ms=5.08115
 tflops=0.422638
 checksum=2.68291e+08
 retained_mantissa_bits=-1 (emulation NOT used / fell back)

[run] mode=lt_64f_emulated_fixedpoint
 avg_ms=0.496538
 tflops=4.32492
 checksum=2.68291e+08
 retained_mantissa_bits=55 (emulation used)

Comparison Summary
------------------
avg_ms (baseline / emu): 5.08115 / 0.496538
tflops (baseline / emu): 0.422638 / 4.32492
checksum delta (emu - baseline): 2.98023e-08
retained bits (baseline / emu): -1 / 55
Speedup (baseline avg_ms / emu avg_ms): 10.2332x
```

- Batched DGEMM with `M=N=K=1024`, `batch_count=4`
```
❯ ./dgemm-lt-fixedpoint-benchmark.x --batched 1 --batch_count 4 --m 1024 --n 1024 --k 1024 --warmup 10 --iters 50
DGEMM fixed-point emulation benchmark via cublasLtMatmul, column-major
M,N,K = 1024,1024,1024 | warmup=10 iters=50
batched=1 batch_count=4
mantissa_control=fixed max_mantissa_bits=55 mantissa_offset=0

GPU: NVIDIA GeForce RTX 3080 (cc 8.6)
cuBLAS version: 130201
Env CUBLAS_EMULATE_DOUBLE_PRECISION = (unset)
Env CUBLAS_EMULATION_STRATEGY = (unset)
Workspace bytes = 207659008 (198.039 MiB)

[algo] mode=lt_64f
 id=22 tile=13 splitK=1 reduction=0 swizzle=0

[algo] mode=lt_64f_emulated_fixedpoint
 id=65 tile=0 splitK=1 reduction=0 swizzle=0

[run] mode=lt_64f
 avg_ms=18.4449
 tflops=0.465708
 checksum=1.07398e+09
 retained_mantissa_bits=-1 (emulation NOT used / fell back)

[run] mode=lt_64f_emulated_fixedpoint
 avg_ms=2.07426
 tflops=4.14121
 checksum=1.07398e+09
 retained_mantissa_bits=55 (emulation used)

Comparison Summary
------------------
avg_ms (baseline / emu): 18.4449 / 2.07426
tflops (baseline / emu): 0.465708 / 4.14121
checksum delta (emu - baseline): -4.76837e-07
retained bits (baseline / emu): -1 / 55
Speedup (baseline avg_ms / emu avg_ms): 8.89229x
```
