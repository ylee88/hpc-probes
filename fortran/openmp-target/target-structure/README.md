# OpenMP Target Structure

This miniapp explores how OpenMP target offloading's parallel structure behaves, focusing on the parallel hierarchy and procedure calls in Fortran.

Three implementations are compared:

1. *Trying* to map outermost parallelism to streaming multiprocessors and innermost to threads. Following the suggestion given in [here](https://forums.developer.nvidia.com/t/nvc-omp-parallel-for-in-declare-target-subroutine-does-not-work/335978).
2. Inlined kernel body with a full OpenMP construct.
3. Fine-grained (“functor-like”) subroutine call with a full OpenMP construct.

The goal is to observe how different kernel structures affect correctness and performance, and whether compilers preserve the intended parallel structure.

## Build & Run

```sh
make
./test.x
```

## Results (using nvfortran 25.1 on RTX 3080)

```
 ----------------------------------------
 GPU offload **with** foo() function call
 ----------------------------------------
 CPU time (s):    5.475265026092529
 GPU time (s):    31.40853500366211
 Speedup     :   0.1743241136670060
 Abs diff    :    0.000000000000000
 PASSED
 -------------------------------------------
 GPU offload **without** foo() function call
 -------------------------------------------
 CPU time (s):    5.475265026092529
 GPU time (s):   0.1138260364532471
 Speedup     :    48.10204410782098
 Abs diff    :    0.000000000000000
 PASSED
 ----------------------------------------------------
 GPU offload **functor-like** foo_ijk() function call
 ----------------------------------------------------
 CPU time (s):    5.475265026092529
 GPU time (s):   0.1148719787597656
 Speedup     :    47.66406120280278
 Abs diff    :    0.000000000000000
 PASSED
```

## Observations

Profiled with

```sh
nsys profile --force-overwrite true --trace=cuda,nvtx --stats=true --output test -- ./test.x
```

The first version (with `foo()` function call) shows poor performance, indicating that the intended parallel structure may not be preserved. The profiling results indicate that the generated kernel is launched with `grid:  <<<32, 1, 1>>>, block: <<<1, 1, 1>>>`, which means only one thread is running the subroutine.

On the other hand, both the second and third versions are launching the kernels with `grid:  <<<524288, 1, 1>>>, block: <<<128, 1, 1>>>` configuration, which exposes more parallelism on the GPU correctly.

### Brief kernel tooltips from `nsys`:

####  First version:
```
nvkernel_MAIN__F1L72_4_
Begins: 37.2439s
Ends: 40.3735s (+3.130 s)
grid:  <<<32, 1, 1>>>
block: <<<1, 1, 1>>>
Launch Type: Regular
Static Shared Memory: 0 bytes
Dynamic Shared Memory: 0 bytes
Registers Per Thread: 86
Local Memory Per Thread: 0 bytes
Local Memory Total: 66,846,720 bytes
Shared Memory executed: 16,384 bytes
Shared Memory Bank Size: 4 B
Theoretical occupancy: 33.3333 %
Launched from thread: 1170534
Latency: ←21.979 μs
Correlation ID: 74
Stream: Stream 13
```

####  Second version:
```
nvkernel_MAIN__F1L121_6_
Begins: 40.4962s
Ends: 40.5076s (+11.434 ms)
grid:  <<<524288, 1, 1>>>
block: <<<128, 1, 1>>>
Launch Type: Regular
Static Shared Memory: 0 bytes
Dynamic Shared Memory: 0 bytes
Registers Per Thread: 30
Local Memory Per Thread: 0 bytes
Local Memory Total: 60,162,048 bytes
Shared Memory executed: 16,384 bytes
Shared Memory Bank Size: 4 B
Theoretical occupancy: 100 %
Launched from thread: 1170534
Latency: ←86.232 μs
Correlation ID: 81
Stream: Stream 13
```

#### Third version:
```
nvkernel_MAIN__F1L179_8_
Begins: 40.7364s
Ends: 40.7479s (+11.527 ms)
grid:  <<<524288, 1, 1>>>
block: <<<128, 1, 1>>>
Launch Type: Regular
Static Shared Memory: 0 bytes
Dynamic Shared Memory: 0 bytes
Registers Per Thread: 58
Local Memory Per Thread: 0 bytes
Local Memory Total: 69,074,944 bytes
Shared Memory executed: 16,384 bytes
Shared Memory Bank Size: 4 B
Theoretical occupancy: 66.6667 %
Launched from thread: 1170534
Latency: ←86.279 μs
Correlation ID: 115
Stream: Stream 13
```
