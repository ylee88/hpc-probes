#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>

#include <algorithm>
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
      << "  --m M --n N --k K                 GEMM sizes (default 1024 1024 1024)\n"
      << "  --iters I                         Timed iterations (default 50)\n"
      << "  --warmup W                        Warmup iterations (default 10)\n"
      << "  --batched 0|1                     Use strided batched cublasLtMatmul (default 0)\n"
      << "  --batch_count B                   Number of GEMMs for batched mode (default 1)\n"
      << "  --workspace_mb MB                 Workspace size for cublasLtMatmul (default auto)\n"
      << "  --mantissa_control dyn|fixed      Emulated-only setting (default fixed)\n"
      << "  --max_mantissa_bits B             Emulated-only setting (default 55)\n"
      << "  --mantissa_offset O               Dynamic-only bias (default 0)\n"
      << "\n"
      << "This program runs two cublasLtMatmul paths in one execution:\n"
      << "  1) computeType = CUBLAS_COMPUTE_64F\n"
      << "  2) computeType = CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT\n";
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

struct RunMetrics {
  double avg_ms;
  double tflops;
  double checksum;
  int32_t mantissa_used;
};

int main(int argc, char** argv) {
  int m = 1024, n = 1024, k = 1024;
  int iters = 50, warmup = 10;
  int batched = 0;
  int batch_count = 1;
  cudaEmulationMantissaControl_t mantissaControl = CUDA_EMULATION_MANTISSA_CONTROL_FIXED;
  int maxMantissaBits = 55;
  int mantissaOffset = 0;
  int workspace_mb = -1;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* opt) {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << opt << "\n";
        std::exit(1);
      }
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
      else {
        std::cerr << "Unknown mantissa_control: " << v << "\n";
        return 1;
      }
    } else if (a == "--max_mantissa_bits") {
      maxMantissaBits = std::stoi(need("--max_mantissa_bits"));
    } else if (a == "--mantissa_offset") {
      mantissaOffset = std::stoi(need("--mantissa_offset"));
    } else if (a == "-h" || a == "--help") {
      usage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown arg: " << a << "\n";
      usage(argv[0]);
      return 1;
    }
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

  std::cout << "DGEMM fixed-point emulation benchmark via cublasLtMatmul, column-major\n";
  std::cout << "M,N,K = " << m << "," << n << "," << k
            << " | warmup=" << warmup << " iters=" << iters << "\n";
  std::cout << "batched=" << batched << " batch_count=" << effective_batch_count << "\n";
  std::cout << "mantissa_control="
            << (mantissaControl == CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC ? "dyn" : "fixed")
            << " max_mantissa_bits=" << maxMantissaBits
            << " mantissa_offset=" << mantissaOffset << "\n"
            << "\n";

  print_device_info();

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

  double* dA = nullptr;
  double* dB = nullptr;
  double* dC = nullptr;
  check_cuda(cudaMalloc((void**)&dA, sizeof(double) * total_a_elems), "cudaMalloc dA");
  check_cuda(cudaMalloc((void**)&dB, sizeof(double) * total_b_elems), "cudaMalloc dB");
  check_cuda(cudaMalloc((void**)&dC, sizeof(double) * total_c_elems), "cudaMalloc dC");
  check_cuda(cudaMemcpy(dA, hA.data(), sizeof(double) * total_a_elems, cudaMemcpyHostToDevice),
             "cudaMemcpy A H2D");
  check_cuda(cudaMemcpy(dB, hB.data(), sizeof(double) * total_b_elems, cudaMemcpyHostToDevice),
             "cudaMemcpy B H2D");
  check_cuda(cudaMemcpy(dC, hC0.data(), sizeof(double) * total_c_elems, cudaMemcpyHostToDevice),
             "cudaMemcpy C0 H2D");

  cublasHandle_t cublas_handle;
  check_cublas(cublasCreate(&cublas_handle), "cublasCreate");
  print_cublas_info(cublas_handle);

  cublasLtHandle_t lt_handle;
  check_cublas(cublasLtCreate(&lt_handle), "cublasLtCreate");

  int32_t* d_mantissa_used = nullptr;
  check_cuda(cudaMalloc((void**)&d_mantissa_used, sizeof(int32_t)), "cudaMalloc d_mantissa_used");
  {
    int32_t init = -1;
    check_cuda(cudaMemcpy(d_mantissa_used, &init, sizeof(int32_t), cudaMemcpyHostToDevice),
               "cudaMemcpy mantissa_used init H2D");
  }

  size_t workspace_bytes = 0;
  if (workspace_mb > 0) {
    workspace_bytes = (size_t)workspace_mb * 1024ull * 1024ull;
  } else {
    workspace_bytes = getFixedPointWorkspaceSizeInBytes(
        m, n, k,
        effective_batch_count,
        false,
        mantissaControl,
        maxMantissaBits);
  }
  void* dWorkspace = nullptr;
  check_cuda(cudaMalloc(&dWorkspace, workspace_bytes), "cudaMalloc workspace");
  std::cout << "Workspace bytes = " << workspace_bytes
            << " (" << (workspace_bytes / (1024.0 * 1024.0)) << " MiB)\n";

  const double alpha = 1.0;
  const double beta = 0.0;
  const int lda = m;
  const int ldb = k;
  const int ldc = m;

  cublasLtMatrixLayout_t layoutA = nullptr;
  cublasLtMatrixLayout_t layoutB = nullptr;
  cublasLtMatrixLayout_t layoutC = nullptr;
  cublasLtMatrixLayout_t layoutD = nullptr;
  check_cublas(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_64F, (uint64_t)m, (uint64_t)k, (int64_t)lda),
               "cublasLtMatrixLayoutCreate A");
  check_cublas(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_64F, (uint64_t)k, (uint64_t)n, (int64_t)ldb),
               "cublasLtMatrixLayoutCreate B");
  check_cublas(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_64F, (uint64_t)m, (uint64_t)n, (int64_t)ldc),
               "cublasLtMatrixLayoutCreate C");
  check_cublas(cublasLtMatrixLayoutCreate(&layoutD, CUDA_R_64F, (uint64_t)m, (uint64_t)n, (int64_t)ldc),
               "cublasLtMatrixLayoutCreate D");

  if (batched) {
    int32_t batchCountLt = effective_batch_count;
    int64_t strideA = (int64_t)a_elems_per_batch;
    int64_t strideB = (int64_t)b_elems_per_batch;
    int64_t strideC = (int64_t)c_elems_per_batch;

    // Layout config: batch count
    check_cublas(cublasLtMatrixLayoutSetAttribute(layoutA,
                                                  CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                  &batchCountLt,
                                                  sizeof(batchCountLt)),
                 "cublasLtMatrixLayoutSetAttribute A batch_count");
    check_cublas(cublasLtMatrixLayoutSetAttribute(layoutB,
                                                  CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                  &batchCountLt,
                                                  sizeof(batchCountLt)),
                 "cublasLtMatrixLayoutSetAttribute B batch_count");
    check_cublas(cublasLtMatrixLayoutSetAttribute(layoutC,
                                                  CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                  &batchCountLt,
                                                  sizeof(batchCountLt)),
                 "cublasLtMatrixLayoutSetAttribute C batch_count");
    check_cublas(cublasLtMatrixLayoutSetAttribute(layoutD,
                                                  CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                  &batchCountLt,
                                                  sizeof(batchCountLt)),
                 "cublasLtMatrixLayoutSetAttribute D batch_count");

    // Layout config: strides
    check_cublas(cublasLtMatrixLayoutSetAttribute(layoutA,
                                                  CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &strideA,
                                                  sizeof(strideA)),
                 "cublasLtMatrixLayoutSetAttribute A stride");
    check_cublas(cublasLtMatrixLayoutSetAttribute(layoutB,
                                                  CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &strideB,
                                                  sizeof(strideB)),
                 "cublasLtMatrixLayoutSetAttribute B stride");
    check_cublas(cublasLtMatrixLayoutSetAttribute(layoutC,
                                                  CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &strideC,
                                                  sizeof(strideC)),
                 "cublasLtMatrixLayoutSetAttribute C stride");
    check_cublas(cublasLtMatrixLayoutSetAttribute(layoutD,
                                                  CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                  &strideC,
                                                  sizeof(strideC)),
                 "cublasLtMatrixLayoutSetAttribute D stride");
  }

  // Create matmul descriptors for both the 64F and emulated cases.
  cublasLtMatmulDesc_t desc64f = nullptr;
  cublasLtMatmulDesc_t descEmu = nullptr;
  check_cublas(cublasLtMatmulDescCreate(&desc64f, CUBLAS_COMPUTE_64F, CUDA_R_64F),
               "cublasLtMatmulDescCreate 64f");
  check_cublas(cublasLtMatmulDescCreate(&descEmu, CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, CUDA_R_64F),
               "cublasLtMatmulDescCreate emu");

  // Common matmul descriptor attributes
  cublasOperation_t trans = CUBLAS_OP_N;
  check_cublas(cublasLtMatmulDescSetAttribute(desc64f,
                                              CUBLASLT_MATMUL_DESC_TRANSA,
                                              &trans,
                                              sizeof(trans)),
               "cublasLtMatmulDescSetAttribute 64f transa");
  check_cublas(cublasLtMatmulDescSetAttribute(desc64f,
                                              CUBLASLT_MATMUL_DESC_TRANSB,
                                              &trans,
                                              sizeof(trans)),
               "cublasLtMatmulDescSetAttribute 64f transb");
  check_cublas(cublasLtMatmulDescSetAttribute(descEmu,
                                              CUBLASLT_MATMUL_DESC_TRANSA,
                                              &trans,
                                              sizeof(trans)),
               "cublasLtMatmulDescSetAttribute emu transa");
  check_cublas(cublasLtMatmulDescSetAttribute(descEmu,
                                              CUBLASLT_MATMUL_DESC_TRANSB,
                                              &trans,
                                              sizeof(trans)),
               "cublasLtMatmulDescSetAttribute emu transb");

  // Create emulation-specific descriptor
  cublasLtEmulationDesc_t emulationDesc = nullptr;
  check_cublas(cublasLtEmulationDescCreate(&emulationDesc), "cublasLtEmulationDescCreate");

  // Configure emulation descriptor
  cublasEmulationStrategy_t strategy = CUBLAS_EMULATION_STRATEGY_EAGER;
  check_cublas(cublasLtEmulationDescSetAttribute(emulationDesc,
                                                 CUBLASLT_EMULATION_DESC_STRATEGY,
                                                 &strategy,
                                                 sizeof(strategy)),
               "cublasLtEmulationDescSetAttribute strategy");
  check_cublas(cublasLtEmulationDescSetAttribute(emulationDesc,
                                                 CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_CONTROL,
                                                 &mantissaControl,
                                                 sizeof(mantissaControl)),
               "cublasLtEmulationDescSetAttribute mantissa_control");
  check_cublas(cublasLtEmulationDescSetAttribute(emulationDesc,
                                                 CUBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT,
                                                 &maxMantissaBits,
                                                 sizeof(maxMantissaBits)),
               "cublasLtEmulationDescSetAttribute max_mantissa_bits");
  if (mantissaControl == CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC) {
    check_cublas(cublasLtEmulationDescSetAttribute(emulationDesc,
                                                   CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_BIT_OFFSET,
                                                   &mantissaOffset,
                                                   sizeof(mantissaOffset)),
                 "cublasLtEmulationDescSetAttribute mantissa_offset");
  }
  check_cublas(cublasLtEmulationDescSetAttribute(emulationDesc,
                                                 CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_BIT_COUNT_POINTER,
                                                 &d_mantissa_used,
                                                 sizeof(d_mantissa_used)),
               "cublasLtEmulationDescSetAttribute mantissa_pointer");

  // Attach emulation descriptor to the emulated matmul descriptor
  check_cublas(cublasLtMatmulDescSetAttribute(descEmu,
                                              CUBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR,
                                              &emulationDesc,
                                              sizeof(emulationDesc)),
               "cublasLtMatmulDescSetAttribute emulation_descriptor");

  // Helper function retrieves the possible algorithms for cublasLtMatmul,
  // given an operation descriptor and matrix layouts.
  auto select_algo = [&](cublasLtMatmulDesc_t opDesc, cublasLtMatmulAlgo_t* algo) -> bool {
    cublasLtMatmulPreference_t pref = nullptr;
    check_cublas(cublasLtMatmulPreferenceCreate(&pref), "cublasLtMatmulPreferenceCreate");
    check_cublas(cublasLtMatmulPreferenceSetAttribute(pref,
                                                      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                      &workspace_bytes,
                                                      sizeof(workspace_bytes)),
                 "cublasLtMatmulPreferenceSetAttribute max_workspace");

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returnAlgoCount = 0;
    cublasStatus_t stat = cublasLtMatmulAlgoGetHeuristic(
        lt_handle,
        opDesc,
        layoutA,
        layoutB,
        layoutC,
        layoutD,
        pref,
        1,   // request the estimated best performing algorithm
        &heuristic,
        &returnAlgoCount);
    check_cublas(cublasLtMatmulPreferenceDestroy(pref), "cublasLtMatmulPreferenceDestroy");

    if (stat != CUBLAS_STATUS_SUCCESS || returnAlgoCount <= 0 || heuristic.state != CUBLAS_STATUS_SUCCESS) {
      return false;
    }

    // found a heuristic algo
    *algo = heuristic.algo;
    return true;
  };

  // Select heuristic algorithms
  cublasLtMatmulAlgo_t algo64f{};
  cublasLtMatmulAlgo_t algoEmu{};
  bool hasAlgo64f = select_algo(desc64f, &algo64f);
  bool hasAlgoEmu = select_algo(descEmu, &algoEmu);
  if (!hasAlgo64f) {
    std::cout << "NOTE: no heuristic algo for 64F path; falling back to internal algo selection.\n";
  }
  if (!hasAlgoEmu) {
    std::cout << "NOTE: no heuristic algo for emulated path; falling back to internal algo selection.\n";
  }

  // Matmul execution helper
  auto do_matmul = [&](cublasLtMatmulDesc_t opDesc, const cublasLtMatmulAlgo_t* algo) {
    check_cublas(cublasLtMatmul(
                    lt_handle,
                    opDesc,
                    &alpha,
                    dA,
                    layoutA,
                    dB,
                    layoutB,
                    &beta,
                    dC,
                    layoutC,
                    dC,
                    layoutD,
                    algo,
                    dWorkspace,
                    workspace_bytes,
                    0),
                "cublasLtMatmul");
  };

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
  check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

  // Runs a single case with given opDesc and algo, and returns the result metrics.
  auto run_case = [&](const char* label, cublasLtMatmulDesc_t opDesc, const cublasLtMatmulAlgo_t* algo) -> RunMetrics {
    check_cuda(cudaMemcpy(dC, hC0.data(), sizeof(double) * total_c_elems, cudaMemcpyHostToDevice),
               "cudaMemcpy reset C H2D");
    {
      int32_t init = -1;
      check_cuda(cudaMemcpy(d_mantissa_used, &init, sizeof(int32_t), cudaMemcpyHostToDevice),
                 "cudaMemcpy mantissa_used reset H2D");
    }

    for (int i = 0; i < warmup; ++i) {
      do_matmul(opDesc, algo);
    }
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after warmup");

    check_cuda(cudaEventRecord(start), "cudaEventRecord start");
    for (int i = 0; i < iters; ++i) {
      do_matmul(opDesc, algo);
    }
    check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    float ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");

    RunMetrics out{};
    out.avg_ms = ms / (double)iters;
    const double flops = 2.0 * (double)m * (double)n * (double)k * (double)effective_batch_count;
    const double tsec = out.avg_ms * 1e-3;
    out.tflops = (flops / tsec) / 1e12;

    check_cuda(cudaMemcpy(&out.mantissa_used, d_mantissa_used, sizeof(int32_t), cudaMemcpyDeviceToHost),
               "cudaMemcpy mantissa_used D2H");

    std::vector<double> hC(total_c_elems);
    check_cuda(cudaMemcpy(hC.data(), dC, sizeof(double) * total_c_elems, cudaMemcpyDeviceToHost),
               "cudaMemcpy C D2H");
    out.checksum = checksum_sum(hC);

    std::cout << "\n[run] mode=" << label << "\n"
              << " avg_ms=" << out.avg_ms << "\n"
              << " tflops=" << out.tflops << "\n"
              << " checksum=" << out.checksum << "\n"
              << " retained_mantissa_bits=" << out.mantissa_used
              << (out.mantissa_used < 0 ? " (emulation NOT used / fell back)" : " (emulation used)")
              << "\n";

    return out;
  };

  // Select the heuristic algorithm if found, otherwise pass nullptr
  // to let cublasLtMatmul internally select the algo.
  const cublasLtMatmulAlgo_t* algo64fPtr = hasAlgo64f ? &algo64f : nullptr;
  const cublasLtMatmulAlgo_t* algoEmuPtr = hasAlgoEmu ? &algoEmu : nullptr;

  // Helper function to print the attributes of a selected algorithm, for debugging/inspection purposes.
  auto print_algo_fingerprint = [&](const char* label, const cublasLtMatmulAlgo_t* algo) {
    if (algo == nullptr) {
      std::cout << "\nWARNING: no selected algo for " << label
                << "; passing nullptr to cublasLtMatmul (internal selection).\n";
      return;
    }

    auto print_attr = [&](cublasLtMatmulAlgoConfigAttributes_t attr, const char* name) {
      int value = -1;
      cublasStatus_t stat = cublasLtMatmulAlgoConfigGetAttribute(
          algo, attr, &value, sizeof(value), nullptr);
      if (stat == CUBLAS_STATUS_SUCCESS) {
        std::cout << " " << name << "=" << value;
      } else {
        std::cout << " " << name << "=n/a";
      }
    };

    std::cout << "\n[algo] mode=" << label << "\n";
    print_attr(CUBLASLT_ALGO_CONFIG_ID, "id");
    print_attr(CUBLASLT_ALGO_CONFIG_TILE_ID, "tile");
    print_attr(CUBLASLT_ALGO_CONFIG_SPLITK_NUM, "splitK");
    print_attr(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, "reduction");
    print_attr(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, "swizzle");
    std::cout << "\n";
  };

  // Print the selected algorithm attributes
  print_algo_fingerprint("lt_64f", algo64fPtr);
  print_algo_fingerprint("lt_64f_emulated_fixedpoint", algoEmuPtr);

  // RUN!
  RunMetrics baseline = run_case("lt_64f", desc64f, algo64fPtr);
  RunMetrics emulated = run_case("lt_64f_emulated_fixedpoint", descEmu, algoEmuPtr);

  const double checksum_delta = emulated.checksum - baseline.checksum;
  const double speedup = (emulated.avg_ms > 0.0) ? (baseline.avg_ms / emulated.avg_ms) : 0.0;

  std::cout << "\nComparison Summary\n";
  std::cout << "------------------\n";
  std::cout << "avg_ms (baseline / emu): " << baseline.avg_ms << " / " << emulated.avg_ms << "\n";
  std::cout << "tflops (baseline / emu): " << baseline.tflops << " / " << emulated.tflops << "\n";
  std::cout << "checksum delta (emu - baseline): " << checksum_delta << "\n";
  std::cout << "retained bits (baseline / emu): "
            << baseline.mantissa_used << " / " << emulated.mantissa_used << "\n";
  std::cout << "Speedup (baseline avg_ms / emu avg_ms): " << speedup << "x\n";

  // Clean up
  check_cuda(cudaEventDestroy(start), "cudaEventDestroy start");
  check_cuda(cudaEventDestroy(stop), "cudaEventDestroy stop");

  check_cublas(cublasLtEmulationDescDestroy(emulationDesc), "cublasLtEmulationDescDestroy");
  check_cublas(cublasLtMatmulDescDestroy(desc64f), "cublasLtMatmulDescDestroy 64f");
  check_cublas(cublasLtMatmulDescDestroy(descEmu), "cublasLtMatmulDescDestroy emu");
  check_cublas(cublasLtMatrixLayoutDestroy(layoutA), "cublasLtMatrixLayoutDestroy A");
  check_cublas(cublasLtMatrixLayoutDestroy(layoutB), "cublasLtMatrixLayoutDestroy B");
  check_cublas(cublasLtMatrixLayoutDestroy(layoutC), "cublasLtMatrixLayoutDestroy C");
  check_cublas(cublasLtMatrixLayoutDestroy(layoutD), "cublasLtMatrixLayoutDestroy D");
  check_cublas(cublasLtDestroy(lt_handle), "cublasLtDestroy");
  check_cublas(cublasDestroy(cublas_handle), "cublasDestroy");

  check_cuda(cudaFree(dWorkspace), "cudaFree workspace");
  check_cuda(cudaFree(d_mantissa_used), "cudaFree d_mantissa_used");
  check_cuda(cudaFree(dA), "cudaFree dA");
  check_cuda(cudaFree(dB), "cudaFree dB");
  check_cuda(cudaFree(dC), "cudaFree dC");

  return 0;
}
