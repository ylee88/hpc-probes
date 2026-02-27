#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(err));
    std::exit(1);
  }
}

static void check_cublas(cublasStatus_t stat, const char* what) {
  if (stat != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuBLAS error (%s): status=%d\n", what, (int)stat);
    std::exit(1);
  }
}

static inline int ceildiv(int a, int b) { return (a + b - 1) / b; }

// Safe-bound workspace sizing from cuBLAS docs (may overestimate).
static size_t getFixedPointWorkspaceSizeInBytes(
    int m, int n, int k, int batchCount, bool isComplex,
    cudaEmulationMantissaControl_t mantissaControl, int maxMantissaBitCount) {

  constexpr double MULTIPLIER = 1.25;

  int mult = isComplex ? 2 : 1;
  int numSlices = ceildiv(maxMantissaBitCount + 1, 8);

  int padded_m = ceildiv(m, 1024) * 1024;
  int padded_n = ceildiv(n, 1024) * 1024;
  int padded_k = ceildiv(k, 128) * 128;
  int num_blocks_k = ceildiv(k, 64);

  size_t gemm_workspace =
      sizeof(int8_t) *
      ((size_t)padded_m * padded_k + (size_t)padded_n * padded_k) * mult *
      (size_t)numSlices;

  gemm_workspace += sizeof(int32_t) * ((size_t)padded_m + (size_t)padded_n) * (size_t)mult;

  if (isComplex) {
    gemm_workspace += sizeof(double) * (size_t)m * (size_t)n * (size_t)mult * (size_t)mult;
  }

  size_t adp_workspace = 0;
  if (mantissaControl == CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC) {
    adp_workspace = sizeof(int32_t) *
                    ((size_t)m * (size_t)num_blocks_k +
                     (size_t)n * (size_t)num_blocks_k +
                     (size_t)m * (size_t)n) *
                    (size_t)mult;
  }

  constexpr size_t CONSTANT_SIZE = 128ull * 1024ull * 1024ull;
  return (size_t)(std::max(gemm_workspace, adp_workspace) * (size_t)batchCount * MULTIPLIER) + CONSTANT_SIZE;
}

static void fill_random(std::vector<double>& x, uint64_t seed) {
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (auto& v : x) v = dist(gen);
}

static double checksum_sum(const std::vector<double>& x) {
  double s = 0.0;
  for (double v : x) s += v;
  return s;
}

static const char* env_or_null(const char* k) {
  const char* v = std::getenv(k);
  return (v && std::strlen(v) > 0) ? v : nullptr;
}

static void usage(const char* prog) {
  std::cerr
      << "Usage: " << prog << " [options]\n"
      << "Options:\n"
      << "  --m M --n N --k K                 GEMM sizes (default 4096 4096 4096)\n"
      << "  --iters I                         Timed iterations (default 50)\n"
      << "  --warmup W                        Warmup iterations (default 10)\n"
      << "  --batched 0|1                     Use cublasDgemmBatched (default 0)\n"
      << "  --batch_count B                   Number of GEMMs for batched mode (default 1)\n"
      << "  --workspace_mb MB                 Workspace size for cublasSetWorkspace (default auto)\n"
      << "  --mantissa_control dyn|fixed      (default fixed)\n"
      << "  --max_mantissa_bits B             (default 55; library default for fixed is 55) \n"
      << "  --mantissa_offset O               Dynamic-only bias (default 0)\n"
      << "\n"
      << "Important:\n"
      << "  You should run with:\n"
      << "    CUBLAS_EMULATE_DOUBLE_PRECISION=1\n"
      << "  Otherwise the library may choose not to emulate the floating point.\n";
}

static void print_device_info() {
  int dev = 0;
  check_cuda(cudaGetDevice(&dev), "cudaGetDevice");
  cudaDeviceProp p{};
  check_cuda(cudaGetDeviceProperties(&p, dev), "cudaGetDeviceProperties");
  std::cout << "GPU: " << p.name << " (cc " << p.major << "." << p.minor << ")\n";
}

static void print_cublas_info(cublasHandle_t h) {
  int ver = 0;
  check_cublas(cublasGetVersion(h, &ver), "cublasGetVersion");
  std::cout << "cuBLAS version: " << ver << "\n";

  const char* v = env_or_null("CUBLAS_EMULATE_DOUBLE_PRECISION");
  std::cout << "Env CUBLAS_EMULATE_DOUBLE_PRECISION = " << (v ? v : "(unset)") << "\n";
  const char* strat = env_or_null("CUBLAS_EMULATION_STRATEGY");
  std::cout << "Env CUBLAS_EMULATION_STRATEGY = " << (strat ? strat : "(unset)") << "\n";
}

int main(int argc, char** argv) {
  int m = 1024, n = 1024, k = 1024;
  int iters = 50, warmup = 10;
  int batched = 0;
  int batch_count = 1;

  // Default: fixed mantissa control with 55 bits (doc default for fixed).
  cudaEmulationMantissaControl_t mantissaControl = CUDA_EMULATION_MANTISSA_CONTROL_FIXED;
  int maxMantissaBits = 55;
  int mantissaOffset = 0;
  int workspace_mb = -1; // auto

  // Parsing command line args
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* opt) {
      if (i + 1 >= argc) { std::cerr << "Missing value for " << opt << "\n"; std::exit(1); }
      return std::string(argv[++i]);
    };

    if (a == "--m") m = std::stoi(need("--m"));
    else if (a == "--n") n = std::stoi(need("--n"));
    else if (a == "--k") k = std::stoi(need("--k"));
    else if (a == "--iters") iters = std::stoi(need("--iters"));
    else if (a == "--warmup") warmup = std::stoi(need("--warmup"));
    else if (a == "--batched") batched = std::stoi(need("--batched"));
    else if (a == "--batch_count") batch_count = std::stoi(need("--batch_count"));
    else if (a == "--workspace_mb") workspace_mb = std::stoi(need("--workspace_mb"));
    else if (a == "--mantissa_control") {
      auto v = need("--mantissa_control");
      if (v == "dyn") mantissaControl = CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC;
      else if (v == "fixed") mantissaControl = CUDA_EMULATION_MANTISSA_CONTROL_FIXED;
      else { std::cerr << "Unknown mantissa_control: " << v << "\n"; return 1; }
    } else if (a == "--max_mantissa_bits") maxMantissaBits = std::stoi(need("--max_mantissa_bits"));
    else if (a == "--mantissa_offset") mantissaOffset = std::stoi(need("--mantissa_offset"));
    else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
    else { std::cerr << "Unknown arg: " << a << "\n"; usage(argv[0]); return 1; }
  }

  if (batched != 0 && batched != 1) {
    std::cerr << "--batched must be 0 or 1\n";
    return 1;
  }
  if (batch_count <= 0) {
    std::cerr << "--batch_count must be > 0\n";
    return 1;
  }
  const int effective_batch_count = batched ? batch_count : 1;

  // Prints the configurations
  std::cout << "DGEMM fixed-point emulation benchmark via cublasGemmEx, column-major\n";
  std::cout << "M,N,K = " << m << "," << n << "," << k
            << " | warmup=" << warmup << " iters=" << iters << "\n";
  std::cout << "batched=" << batched << " batch_count=" << effective_batch_count << "\n";
  std::cout << "mantissa_control="
            << (mantissaControl == CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC ? "dyn" : "fixed")
            << " max_mantissa_bits=" << maxMantissaBits
            << " mantissa_offset=" << mantissaOffset << "\n";

  print_device_info();

  // Host buffers
  const size_t a_elems_per_batch = (size_t)m * (size_t)k;
  const size_t b_elems_per_batch = (size_t)k * (size_t)n;
  const size_t c_elems_per_batch = (size_t)m * (size_t)n;
  const size_t total_a_elems = a_elems_per_batch * (size_t)effective_batch_count;
  const size_t total_b_elems = b_elems_per_batch * (size_t)effective_batch_count;
  const size_t total_c_elems = c_elems_per_batch * (size_t)effective_batch_count;

  std::vector<double> hA(total_a_elems), hB(total_b_elems);
  std::vector<double> hC0(total_c_elems, 0.0);
  fill_random(hA, 1234);
  fill_random(hB, 5678);

  // Device buffers
  double *dA = nullptr, *dB = nullptr, *dC = nullptr;
  check_cuda(cudaMalloc((void**)&dA, sizeof(double) * total_a_elems), "cudaMalloc dA");
  check_cuda(cudaMalloc((void**)&dB, sizeof(double) * total_b_elems), "cudaMalloc dB");
  check_cuda(cudaMalloc((void**)&dC, sizeof(double) * total_c_elems), "cudaMalloc dC");
  check_cuda(cudaMemcpy(dA, hA.data(), sizeof(double) * total_a_elems, cudaMemcpyHostToDevice),
             "cudaMemcpy A H2D");
  check_cuda(cudaMemcpy(dB, hB.data(), sizeof(double) * total_b_elems, cudaMemcpyHostToDevice),
             "cudaMemcpy B H2D");
  check_cuda(cudaMemcpy(dC, hC0.data(), sizeof(double) * total_c_elems, cudaMemcpyHostToDevice),
             "cudaMemcpy C0 H2D");

  double** dAarray = nullptr;
  double** dBarray = nullptr;
  double** dCarray = nullptr;
  if (batched) {
    std::vector<double*> hAarray((size_t)effective_batch_count);
    std::vector<double*> hBarray((size_t)effective_batch_count);
    std::vector<double*> hCarray((size_t)effective_batch_count);
    for (int i = 0; i < effective_batch_count; ++i) {
      hAarray[(size_t)i] = dA + ((size_t)i * a_elems_per_batch);
      hBarray[(size_t)i] = dB + ((size_t)i * b_elems_per_batch);
      hCarray[(size_t)i] = dC + ((size_t)i * c_elems_per_batch);
    }

    check_cuda(cudaMalloc((void**)&dAarray, sizeof(double*) * (size_t)effective_batch_count),
               "cudaMalloc dAarray");
    check_cuda(cudaMalloc((void**)&dBarray, sizeof(double*) * (size_t)effective_batch_count),
               "cudaMalloc dBarray");
    check_cuda(cudaMalloc((void**)&dCarray, sizeof(double*) * (size_t)effective_batch_count),
               "cudaMalloc dCarray");

    check_cuda(cudaMemcpy(
                   dAarray,
                   hAarray.data(),
                   sizeof(double*) * (size_t)effective_batch_count,
                   cudaMemcpyHostToDevice),
               "cudaMemcpy Aarray H2D");
    check_cuda(cudaMemcpy(
                   dBarray,
                   hBarray.data(),
                   sizeof(double*) * (size_t)effective_batch_count,
                   cudaMemcpyHostToDevice),
               "cudaMemcpy Barray H2D");
    check_cuda(cudaMemcpy(
                   dCarray,
                   hCarray.data(),
                   sizeof(double*) * (size_t)effective_batch_count,
                   cudaMemcpyHostToDevice),
               "cudaMemcpy Carray H2D");
  }

  cublasHandle_t handle;
  check_cublas(cublasCreate(&handle), "cublasCreate");
  print_cublas_info(handle);

  // Configure fixed-point emulation parameters (mantissa control + limits).
  check_cublas(cublasSetFixedPointEmulationMantissaControl(handle, mantissaControl),
               "cublasSetFixedPointEmulationMantissaControl");
  check_cublas(cublasSetFixedPointEmulationMaxMantissaBitCount(handle, maxMantissaBits),
               "cublasSetFixedPointEmulationMaxMantissaBitCount");
  if (mantissaControl == CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC) {
    check_cublas(cublasSetFixedPointEmulationMantissaBitOffset(handle, mantissaOffset),
                 "cublasSetFixedPointEmulationMantissaBitOffset");
  }

  // Allocate a device int32 and ask cuBLAS to write the *retained mantissa bit count* into it.
  // If emulation is not used (fallback), this will read back as -1.
  int32_t* d_mantissa_used = nullptr;
  check_cuda(cudaMalloc((void**)&d_mantissa_used, sizeof(int32_t)), "cudaMalloc d_mantissa_used");
  {
    int32_t init = -1;
    check_cuda(cudaMemcpy(d_mantissa_used, &init, sizeof(int32_t), cudaMemcpyHostToDevice),
               "cudaMemcpy mantissa_used init H2D");
  }
  check_cublas(cublasSetFixedPointEmulationMantissaBitCountPointer(handle, d_mantissa_used),
               "cublasSetFixedPointEmulationMantissaBitCountPointer");

  // Workspace (avoid cudaMallocAsync internal allocation failures causing fallbacks).
  size_t workspace_bytes = 0;
  if (workspace_mb > 0) {
    workspace_bytes = (size_t)workspace_mb * 1024ull * 1024ull;
  } else {
    workspace_bytes = getFixedPointWorkspaceSizeInBytes(
        m, n, k,
        /*batchCount*/ effective_batch_count,
        /*isComplex*/ false,
        mantissaControl,
        maxMantissaBits);
  }
  void* dWorkspace = nullptr;
  check_cuda(cudaMalloc(&dWorkspace, workspace_bytes), "cudaMalloc workspace");
  check_cublas(cublasSetWorkspace(handle, dWorkspace, workspace_bytes), "cublasSetWorkspace");
  std::cout << "Workspace bytes = " << workspace_bytes
            << " (" << (workspace_bytes / (1024.0 * 1024.0)) << " MiB)\n";

  const double alpha = 1.0;
  const double beta = 0.0;
  const int lda = m, ldb = k, ldc = m;

  // Helper function to run a single GEMM with a given compute type.
  auto do_gemm = [&](cublasComputeType_t computeType) {
    (void)computeType;
    if (batched) {
      check_cublas(
          cublasDgemmBatched(
              handle,
              CUBLAS_OP_N, CUBLAS_OP_N,
              m, n, k,
              &alpha,
              (const double**)dAarray, lda,
              (const double**)dBarray, ldb,
              &beta,
              dCarray, ldc,
              effective_batch_count),
          "cublasDgemmBatched");
    } else {
      // check_cublas(
      //     cublasGemmEx(
      //         handle,
      //         CUBLAS_OP_N, CUBLAS_OP_N,
      //         m, n, k,
      //         &alpha,
      //         dA, CUDA_R_64F, lda,
      //         dB, CUDA_R_64F, ldb,
      //         &beta,
      //         dC, CUDA_R_64F, ldc,
      //         computeType,
      //         CUBLAS_GEMM_DEFAULT),
      //     "cublasGemmEx");
      check_cublas(
          cublasDgemm(
              handle,
              CUBLAS_OP_N, CUBLAS_OP_N,
              m, n, k,
              &alpha,
              dA, lda,
              dB, ldb,
              &beta,
              dC, ldc),
          "cublasDgemm");
    }
  };

  cudaEvent_t start, stop;
  check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
  check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

  // Intentionally hard-code the compute type to the fixed-point emulated FP64 mode.
  // This keeps the benchmark path stable across runs, so users can compare behavior by
  // toggling env vars (for example CUBLAS_EMULATE_DOUBLE_PRECISION) outside the binary.
  // Validate emulation engagement via retained_mantissa_bits below.
  constexpr cublasComputeType_t COMPUTE_TYPE = CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT;

  check_cuda(cudaMemcpy(dC, hC0.data(), sizeof(double) * total_c_elems, cudaMemcpyHostToDevice),
             "cudaMemcpy reset C H2D");
  {
    int32_t init = -1;
    check_cuda(cudaMemcpy(d_mantissa_used, &init, sizeof(int32_t), cudaMemcpyHostToDevice),
               "cudaMemcpy mantissa_used reset H2D");
  }

  // Warmup
  for (int i = 0; i < warmup; ++i) do_gemm(COMPUTE_TYPE);
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after warmup");

  // Timed runs
  check_cuda(cudaEventRecord(start), "cudaEventRecord start");
  for (int i = 0; i < iters; ++i) do_gemm(COMPUTE_TYPE);
  check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
  check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

  float ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");

  double avg_ms = ms / (double)iters;
  double flops = 2.0 * (double)m * (double)n * (double)k * (double)effective_batch_count;
  double tsec = avg_ms * 1e-3;
  double tflops = (flops / tsec) / 1e12;
  int32_t mantissa_used = -999;
  check_cuda(cudaMemcpy(&mantissa_used, d_mantissa_used, sizeof(int32_t), cudaMemcpyDeviceToHost),
             "cudaMemcpy mantissa_used D2H");

  std::vector<double> hC(total_c_elems);
  check_cuda(cudaMemcpy(hC.data(), dC, sizeof(double) * total_c_elems, cudaMemcpyDeviceToHost),
             "cudaMemcpy C D2H");
  double checksum = checksum_sum(hC);

  std::cout << "[run] avg_ms=" << avg_ms
            << " tflops=" << tflops
            << " checksum=" << checksum
            << " retained_mantissa_bits=" << mantissa_used
            << (mantissa_used < 0 ? " (emulation NOT used / fell back)" : " (emulation used)")
            << "\n";

  if (mantissa_used < 0) {
    std::cout
        << "NOTE: retained bits == -1 usually indicates fallback to non-emulated routines.\n";
  }

  // Cleanup
  check_cublas(cublasDestroy(handle), "cublasDestroy");
  if (dAarray) check_cuda(cudaFree(dAarray), "cudaFree dAarray");
  if (dBarray) check_cuda(cudaFree(dBarray), "cudaFree dBarray");
  if (dCarray) check_cuda(cudaFree(dCarray), "cudaFree dCarray");
  check_cuda(cudaFree(dWorkspace), "cudaFree workspace");
  check_cuda(cudaFree(d_mantissa_used), "cudaFree d_mantissa_used");
  check_cuda(cudaFree(dA), "cudaFree dA");
  check_cuda(cudaFree(dB), "cudaFree dB");
  check_cuda(cudaFree(dC), "cudaFree dC");
  check_cuda(cudaEventDestroy(start), "cudaEventDestroy start");
  check_cuda(cudaEventDestroy(stop), "cudaEventDestroy stop");
  return 0;
}
