# OpenACC Target Structure

This is a mirror to [OpenMP Target Structure](../../openmp-target/target-structure), but for OpenACC.

## Build & Run

```sh
make
./test.x
```

## Results (using nvfortran 25.11 on RTX 3080)

```
 ----------------------------------------
 GPU offload **with** foo() function call
 ----------------------------------------
 CPU time (s):    5.445108175277710
 GPU time (s):    1.071078062057495
 Speedup     :    5.083764076745154
 Abs diff    :    0.000000000000000
 PASSED
 -------------------------------------------
 GPU offload **without** foo() function call
 -------------------------------------------
 CPU time (s):    5.445108175277710
 GPU time (s):   0.1157929897308350
 Speedup     :    47.02450630159100
 Abs diff    :    0.000000000000000
 PASSED
 ----------------------------------------------------
 GPU offload **functor-like** foo_ijk() function call
 ----------------------------------------------------
 CPU time (s):    5.445108175277710
 GPU time (s):   0.1193408966064453
 Speedup     :    45.62650633700395
 Abs diff    :    0.000000000000000
 PASSED
```

## Observations

Profiled with

```sh
nsys profile --force-overwrite true --trace=cuda,nvtx --stats=true --output test -- ./test.x
```

### Brief kernel tooltips from `nsys`:

#### First version:
```
main_65_gpu
Begins: 5.87548s
Ends: 5.98784s (+112.365 ms)
grid:  <<<32, 1, 1>>>
block: <<<32, 1, 1>>>
Launch Type: Regular
Static Shared Memory: 0 bytes
Dynamic Shared Memory: 0 bytes
Registers Per Thread: 72
Local Memory Per Thread: 0 bytes
Local Memory Total: 80,216,064 bytes
Shared Memory executed: 16,384 bytes
Shared Memory Bank Size: 4 B
Theoretical occupancy: 33.3333 %
Cluster X: 0
Cluster Y: 0
Cluster Z: 0
Cluster Scheduling Policy: 0
Max Potential Cluster Size: 0
Max Active Clusters: 0
Launched from thread: 308014
Latency: ←22.719 μs
Correlation ID: 46
Stream: Stream 13
```

#### Second version:
```
main_121_gpu
Begins: 7.17211s
Ends: 7.18329s (+11.176 ms)
grid:  <<<524288, 1, 1>>>
block: <<<128, 1, 1>>>
Launch Type: Regular
Static Shared Memory: 0 bytes
Dynamic Shared Memory: 0 bytes
Registers Per Thread: 16
Local Memory Per Thread: 0 bytes
Local Memory Total: 60,162,048 bytes
Shared Memory executed: 16,384 bytes
Shared Memory Bank Size: 4 B
Theoretical occupancy: 100 %
Cluster X: 0
Cluster Y: 0
Cluster Z: 0
Cluster Scheduling Policy: 0
Max Potential Cluster Size: 0
Max Active Clusters: 0
Launched from thread: 308014
Latency: ←86.992 μs
Correlation ID: 94
Stream: Stream 13
```

#### Third version:
```
main_179_gpu
Begins: 7.41275s
Ends: 7.42444s (+11.689 ms)
grid:  <<<524288, 1, 1>>>
block: <<<128, 1, 1>>>
Launch Type: Regular
Static Shared Memory: 20 bytes
Dynamic Shared Memory: 0 bytes
Registers Per Thread: 46
Local Memory Per Thread: 0 bytes
Local Memory Total: 69,074,944 bytes
Shared Memory executed: 16,384 bytes
Shared Memory Bank Size: 4 B
Theoretical occupancy: 83.3333 %
Cluster X: 0
Cluster Y: 0
Cluster Z: 0
Cluster Scheduling Policy: 0
Max Potential Cluster Size: 0
Max Active Clusters: 0
Launched from thread: 308014
Latency: ←115.395 μs
Correlation ID: 138
Stream: Stream 13
```
