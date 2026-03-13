# cuBLAS floating point emulation (Ozaki Scheme) test for DGEMM

This is a test for cuBLAS floating point emulation (Ozaki Scheme) for DGEMM. The test compares the performance and checksum of the cuBLAS implementation with and without FP emulation.

Reference: https://docs.nvidia.com/cuda/cublas/index.html#fixed-point

## Build & Run

```sh
make
./run_compare_results.py
```

## Results (using nvcc 13.1 on RTX 3080)

```
== Mode A: env vars unset (baseline) ==
DGEMM fixed-point emulation benchmark via cublasGemmEx, column-major
M,N,K = 1024,1024,1024 | warmup=10 iters=50
mantissa_control=fixed max_mantissa_bits=55 mantissa_offset=0
GPU: NVIDIA GeForce RTX 3080 (cc 8.6)
cuBLAS version: 130201
Env CUBLAS_EMULATE_DOUBLE_PRECISION = (unset)
Env CUBLAS_EMULATION_STRATEGY = (unset)
Workspace bytes = 152578048 (145.51 MiB)
[run] avg_ms=5.17552 tflops=0.414931 checksum=2.68291e+08 retained_mantissa_bits=-1 (emulation NOT used / fell back)
NOTE: retained bits == -1 usually indicates fallback to non-emulated routines.

== Mode B: fixed-point emulation enabled via env vars ==
DGEMM fixed-point emulation benchmark via cublasGemmEx, column-major
M,N,K = 1024,1024,1024 | warmup=10 iters=50
mantissa_control=fixed max_mantissa_bits=55 mantissa_offset=0
GPU: NVIDIA GeForce RTX 3080 (cc 8.6)
cuBLAS version: 130201
Env CUBLAS_EMULATE_DOUBLE_PRECISION = 1
Env CUBLAS_EMULATION_STRATEGY = eager
Workspace bytes = 152578048 (145.51 MiB)
[run] avg_ms=0.494182 tflops=4.34553 checksum=2.68291e+08 retained_mantissa_bits=55 (emulation used)

Comparison Summary
------------------
Metric                             Mode A           Mode B       B vs A
avg_ms (lower better)             5.17552         0.494182       0.095x
tflops (higher better)           0.414931          4.34553      10.473x
checksum                      2.68291e+08      2.68291e+08            0
retained_bits                          -1               55          n/a

Speedup (Mode A / Mode B): 10.4729x
Mode B emulation status: engaged
```
