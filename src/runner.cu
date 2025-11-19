#include "kernels.cuh"
#include "runner.cuh"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <cuda_bf16.h> 

float get_sec() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return (1e6 * time.tv_sec + time.tv_usec);
}

float cpu_elapsed_time(float &beg, float &end) { return 1.0e-6 * (end - beg); }

void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};

void CudaDeviceInfo() {
  int deviceId;

  cudaGetDevice(&deviceId);

  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, deviceId);

  printf("Device ID: %d\n\
    Name: %s\n\
    Compute Capability: %d.%d\n\
    memoryBusWidth: %d\n\
    maxThreadsPerBlock: %d\n\
    maxThreadsPerMultiProcessor: %d\n\
    maxRegsPerBlock: %d\n\
    maxRegsPerMultiProcessor: %d\n\
    totalGlobalMem: %zuMB\n\
    sharedMemPerBlock: %zuKB\n\
    sharedMemPerMultiprocessor: %zuKB\n\
    totalConstMem: %zuKB\n\
    multiProcessorCount: %d\n\
    Warp Size: %d\n",
         deviceId, props.name, props.major, props.minor, props.memoryBusWidth,
         props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
         props.regsPerBlock, props.regsPerMultiprocessor,
         props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
         props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
         props.multiProcessorCount, props.warpSize);
};

void randomize_matrix(float *mat, int N) {
  // NOTICE: Use gettimeofday instead of srand((unsigned)time(NULL)); the time
  // precision is too low and the same random number is generated.
  struct timeval time {};
  gettimeofday(&time, nullptr);
  srand(time.tv_usec);
  // srand(0);
  for (int i = 0; i < N; i++) {
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
    mat[i] = tmp;
  }
}

void range_init_matrix(float *mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = i;
  }
}

void zero_init_matrix(float *mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = 0.0;
  }
}

void copy_matrix(const float *src, float *dest, int N) {
  int i;
  for (i = 0; src + i && dest + i && i < N; i++)
    *(dest + i) = *(src + i);
  if (i != N)
    printf("copy failed at %d while there are %d elements in total.\n", i, N);
}

void print_matrix(const float *A, int M, int N, std::ofstream &fs) {
  int i;
  fs << std::setprecision(2)
     << std::fixed; // Set floating-point precision and fixed notation
  fs << "[";
  for (i = 0; i < M * N; i++) {
    if ((i + 1) % N == 0)
      fs << std::setw(5) << A[i]; // Set field width and write the value
    else
      fs << std::setw(5) << A[i] << ", ";
    if ((i + 1) % N == 0) {
      if (i + 1 < M * N)
        fs << ";\n";
    }
  }
  fs << "]\n";
}

bool verify_matrix(float *matRef, float *matOut, int N, int kernel_num) {
  double diff = 0.0;
  double threshold = 0.01;
  if (kernel_num > 29) {
    // bf16 warptiling needs a higher threshold
    threshold = 4.0f;
  }
  int i;
  for (i = 0; i < N; i++) {
    diff = std::fabs(matRef[i] - matOut[i]);
    if (isnan(diff) || diff > threshold) {
      printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
             matRef[i], matOut[i], diff, i);
      return false;
    }
  }
  return true;
}

int div_ceil(int numerator, int denominator) {
  std::div_t res = std::div(numerator, denominator);
  return res.rem ? (res.quot + 1) : res.quot;
}

void runCublasFP32(cublasHandle_t handle, int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C) {
  // cuBLAS uses column-major order. So we change the order of our row-major A &
  // B, since (B^T*A^T)^T = (A*B)
  // This runs cuBLAS in full fp32 mode
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
               N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void runCublasBF16(cublasHandle_t handle, int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C) {
  // This runs cuBLAS with mixed precision (performing the mul with operands
  // downcast to bf16), which is ~4x faster
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
               N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N,
               CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void runCublasBF16_AB(cublasHandle_t handle, int M, int N, int K, float alpha,
                   __nv_bfloat16 *A, __nv_bfloat16 *B, float beta, float *C) {
  // This runs cuBLAS with mixed precision (performing the mul with operands
  // downcast to bf16), which is ~4x faster
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16BF,
               N, A, CUDA_R_16BF, K, &beta, C, CUDA_R_32F, N,
               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void runCublasTF32(cublasHandle_t handle, int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C) {
  // This runs cuBLAS with mixed precision (performing the mul with operands
  // downcast to bf16), which is ~4x faster
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
               N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N,
               CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void run_sgemm_naive(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32, 32);
  sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_my_sgemm_naive(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32, 32);
  my_sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_coalesce(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32 * 32);
  sgemm_global_mem_coalesce<32>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_my_sgemm_coalesce(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32, 32);
  my_sgemm_global_mem_coalesce<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_shared_mem_block(int M, int N, int K, float alpha, float *A,
                                float *B, float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32 * 32);
  // L1 cache becomes useless, since we access GMEM only via SMEM, so we carve
  // out all of L1 to SMEM. This doesn't currently make a difference, since
  // occupancy is limited by reg and thread count, but it's good to do anyway.
  cudaFuncSetAttribute(sgemm_shared_mem_block<32>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  sgemm_shared_mem_block<32>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_my_sgemm_shared_mem_block(int M, int N, int K, float alpha, float *A,
                                float *B, float beta, float *C) {
  dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
  dim3 blockDim(32, 32);
  // L1 cache becomes useless, since we access GMEM only via SMEM, so we carve
  // out all of L1 to SMEM. This doesn't currently make a difference, since
  // occupancy is limited by reg and thread count, but it's good to do anyway.
  cudaFuncSetAttribute(my_sgemm_shared_mem_block<32>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  my_sgemm_shared_mem_block<32>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runSgemm1DBlocktiling(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  const uint BM = 64;
  const uint BN = 64;
  const uint BK = 8;
  const uint TM = 8;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / TM);
  sgemm1DBlocktiling<BM, BN, BK, TM>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_my_Sgemm1DBlocktiling(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  const uint BM = 64;
  const uint BN = 64;
  const uint BK = 8;
  const uint TM = 8;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim(BM, BN / TM);
  my_sgemm1DBlocktiling<BM, BN, BK, TM>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runSgemm2DBlocktiling(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemm2DBlocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    // this is a hacky solution to the underlying problem
    // of not having proper bounds checking in the kernel
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemm2DBlocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}

void run_my_Sgemm2DBlocktiling(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  const uint BM = 128;
  const uint BN = 128;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim(BM / TM, BN / TN);
  my_sgemm2DBlocktiling<BM, BN, BK, TM, TN>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  
}

void run_my_Sgemm2DBlocktiling_v2(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  const uint BM = 128;
  const uint BN = 128;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim(BN / TN, BM / TM);
  my_sgemm2DBlocktiling_v2<BM, BN, BK, TM, TN>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  
}

void run_my_Sgemm2DBlocktiling_v3(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  const uint BM = 128;
  const uint BN = 128;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim(BN / TN, BM / TM);
  my_sgemm2DBlocktiling_v3<BM, BN, BK, TM, TN>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  
}

void runSgemmVectorize(int M, int N, int K, float alpha, float *A, float *B,
                       float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmVectorize<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    // this is a hacky solution to the underlying problem
    // of not having proper bounds checking in the kernel
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmVectorize<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}

void run_my_SgemmVectorize(int M, int N, int K, float alpha, float *A, float *B,
                       float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
 
  const uint BM = 128;
  const uint BN = 128;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim(BN / TN, BM / TM);
  my_sgemmVectorize<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runSgemmResolveBankConflicts(int M, int N, int K, float alpha, float *A,
                                  float *B, float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmResolveBankConflicts<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    // this is a hacky solution to the underlying problem
    // of not having proper bounds checking in the kernel
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmResolveBankConflicts<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}

void run_my_ResolveBankConflicts(int M, int N, int K, float alpha, float *A, float *B,
                       float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
 
  const uint BM = 128;
  const uint BN = 128;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim(BN / TN, BM / TM);
  my_sgemmResolveBankConflicts<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runSgemmResolveBankExtraCol(int M, int N, int K, float alpha, float *A,
                                 float *B, float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmResolveBankExtraCol<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    // this is a hacky solution to the underlying problem
    // of not having proper bounds checking in the kernel
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmResolveBankExtraCol<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}

void runSgemmAutotuned(int M, int N, int K, float alpha, float *A, float *B,
                       float beta, float *C) {
  // A100
  // const uint K9_BK = 16;
  // const uint K9_TM = 4;
  // const uint K9_TN = 4;
  // const uint K9_BM = 64;
  // const uint K9_BN = 64;
  // A6000
  const uint K9_BK = 16;
  const uint K9_TM = 8;
  const uint K9_TN = 8;
  const uint K9_BM = 128;
  const uint K9_BN = 128;
  dim3 blockDim(K9_NUM_THREADS);

  static_assert(
      (K9_NUM_THREADS * 4) % K9_BK == 0,
      "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization issues "
      "during GMEM->SMEM tiling (loading only parts of the final row of Bs "
      "during each iteraion)");
  static_assert(
      (K9_NUM_THREADS * 4) % K9_BN == 0,
      "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization issues "
      "during GMEM->SMEM tiling (loading only parts of the final row of As "
      "during each iteration)");
  static_assert(
      K9_BN % (16 * K9_TN) == 0,
      "K9_BN must be a multiple of 16*K9_TN to avoid quantization effects");
  static_assert(
      K9_BM % (16 * K9_TM) == 0,
      "K9_BM must be a multiple of 16*K9_TM to avoid quantization effects");
  static_assert((K9_BM * K9_BK) % (4 * K9_NUM_THREADS) == 0,
                "K9_BM*K9_BK must be a multiple of 4*256 to vectorize loads");
  static_assert((K9_BN * K9_BK) % (4 * K9_NUM_THREADS) == 0,
                "K9_BN*K9_BK must be a multiple of 4*256 to vectorize loads");

  dim3 gridDim(CEIL_DIV(N, K9_BN), CEIL_DIV(M, K9_BM));
  sgemmAutotuned<K9_BM, K9_BN, K9_BK, K9_TM, K9_TN>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

// 这是 Kernel 20 的 autotuning host 函数
void runKernel20Autotuned(int M, int N, int K, float alpha, float *A, float *B,
                         float beta, float *C) {
  // =================================================================
  // 这些常量将会被 autotune_kernel20.sh 脚本自动修改
  // 这里的初始值只是一个占位符
  const uint K20_BK = 32;
  const uint K20_TM = 8;
  const uint K20_TN = 8;
  const uint K20_BM = 128;
  const uint K20_BN = 128;
  // =================================================================
  
  // blockDim 的计算方式与你的原始代码保持一致
  const uint K20_BLOCK_DIM_X = K20_BN / K20_TN;
  const uint K20_BLOCK_DIM_Y = K20_BM / K20_TM;
  const uint K20_NUM_THREADS = K20_BLOCK_DIM_X * K20_BLOCK_DIM_Y;

  // 添加 static_assert 来在编译时检查约束条件
  // 这样如果脚本生成了无效的参数组合，编译会直接失败，节省调试时间
  
  // 约束1：确保 blockDim 可以整除
  static_assert(K20_BN % K20_TN == 0, "K20_BN must be a multiple of K20_TN");
  static_assert(K20_BM % K20_TM == 0, "K20_BM must be a multiple of K20_TM");

  // 约束2：确保线程总数不超过 GPU 限制 (通常是 1024)
  static_assert(K20_NUM_THREADS <= 1024, "Total number of threads exceeds 1024");
  
  // 约束3：As tile 的大小必须能被线程块的总加载量整除 (float4 vectorized load)
  // 每个线程加载 (BM*BK)/(NT) 个元素, 每次加载4个, 所以是 (BM*BK)/(NT*4) 次
  static_assert((K20_BM * K20_BK) % (4 * K20_NUM_THREADS) == 0,
                "The total elements in As tile (BM*BK) must be divisible by "
                "the total float4 load capacity of the block (4 * NUM_THREADS)");

  // 约束4：Bs tile 的大小必须能被线程块的总加载量整除
  static_assert((K20_BN * K20_BK) % (4 * K20_NUM_THREADS) == 0,
                "The total elements in Bs tile (BN*BK) must be divisible by "
                "the total float4 load capacity of the block (4 * NUM_THREADS)");

  // 设置 Grid 和 Block 维度
  dim3 gridDim(CEIL_DIV(N, K20_BN), CEIL_DIV(M, K20_BM));
  dim3 blockDim(K20_BLOCK_DIM_X, K20_BLOCK_DIM_Y);

  // 启动你的 Kernel 20
  my_sgemmVectorize<K20_BM, K20_BN, K20_BK, K20_TM, K20_TN>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runSgemmWarptiling(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  // Settings for A100
  // const uint K10_NUM_THREADS = 128;
  // const uint K10_BN = 128;
  // const uint K10_BM = 64;
  // const uint K10_BK = 16;
  // const uint K10_WN = 64;
  // const uint K10_WM = 32;
  // const uint K10_WNITER = 1;
  // const uint K10_TN = 4;
  // const uint K10_TM = 4;
  // Settings for A6000
  const uint K10_NUM_THREADS = 128;
  const uint K10_BN = 128;
  const uint K10_BM = 128;
  const uint K10_BK = 16;
  const uint K10_WN = 64;
  const uint K10_WM = 64;
  const uint K10_WNITER = 4;
  const uint K10_TN = 4;
  const uint K10_TM = 8;
  dim3 blockDim(K10_NUM_THREADS);

  constexpr uint NUM_WARPS = K10_NUM_THREADS / 32;

  // warptile in threadblocktile
  static_assert((K10_BN % K10_WN == 0) and (K10_BM % K10_WM == 0));
  static_assert((K10_BN / K10_WN) * (K10_BM / K10_WM) == NUM_WARPS);

  // threads in warpsubtile
  static_assert((K10_WM * K10_WN) % (WARPSIZE * K10_TM * K10_TN * K10_WNITER) ==
                0);
  constexpr uint K10_WMITER =
      (K10_WM * K10_WN) / (32 * K10_TM * K10_TN * K10_WNITER);
  // warpsubtile in warptile
  static_assert((K10_WM % K10_WMITER == 0) and (K10_WN % K10_WNITER == 0));

  static_assert((K10_NUM_THREADS * 4) % K10_BK == 0,
                "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of Bs during each iteraion)");
  static_assert((K10_NUM_THREADS * 4) % K10_BN == 0,
                "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of As during each iteration)");
  static_assert(K10_BN % (16 * K10_TN) == 0,
                "BN must be a multiple of 16*TN to avoid quantization effects");
  static_assert(K10_BM % (16 * K10_TM) == 0,
                "BM must be a multiple of 16*TM to avoid quantization effects");
  static_assert((K10_BM * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                "BM*BK must be a multiple of 4*256 to vectorize loads");
  static_assert((K10_BN * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                "BN*BK must be a multiple of 4*256 to vectorize loads");

  dim3 gridDim(CEIL_DIV(N, K10_BN), CEIL_DIV(M, K10_BM));
  sgemmWarptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM,
                  K10_TN, K10_NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_my_SgemmWarptiling(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  // Settings for A100
  // const uint K23_NUM_THREADS = 128;
  // const uint K23_BN = 128;
  // const uint K23_BM = 64;
  // const uint K23_BK = 16;
  // const uint K23_WN = 64;
  // const uint K23_WM = 32;
  // const uint K23_WNITER = 1;
  // const uint K23_TN = 4;
  // const uint K23_TM = 4;
  // Settings for A6000
  const uint K23_NUM_THREADS = 128;
  const uint K23_BN = 128;
  const uint K23_BM = 128;
  const uint K23_BK = 16;
  const uint K23_WN = 64;
  const uint K23_WM = 64;
  const uint K23_WNITER = 4;
  const uint K23_TN = 4;
  const uint K23_TM = 8;
  dim3 blockDim(K23_NUM_THREADS);

  constexpr uint NUM_WARPS = K23_NUM_THREADS / 32;

  // warptile in threadblocktile
  static_assert((K23_BN % K23_WN == 0) and (K23_BM % K23_WM == 0));
  static_assert((K23_BN / K23_WN) * (K23_BM / K23_WM) == NUM_WARPS);

  // threads in warpsubtile
  static_assert((K23_WM * K23_WN) % (WARPSIZE * K23_TM * K23_TN * K23_WNITER) ==
                0);
  constexpr uint K23_WMITER =
      (K23_WM * K23_WN) / (32 * K23_TM * K23_TN * K23_WNITER);
  // warpsubtile in warptile
  static_assert((K23_WM % K23_WMITER == 0) and (K23_WN % K23_WNITER == 0));

  static_assert((K23_NUM_THREADS * 4) % K23_BK == 0,
                "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of Bs during each iteraion)");
  static_assert((K23_NUM_THREADS * 4) % K23_BN == 0,
                "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of As during each iteration)");
  static_assert(K23_BN % (16 * K23_TN) == 0,
                "BN must be a multiple of 16*TN to avoid quantization effects");
  static_assert(K23_BM % (16 * K23_TM) == 0,
                "BM must be a multiple of 16*TM to avoid quantization effects");
  static_assert((K23_BM * K23_BK) % (4 * K23_NUM_THREADS) == 0,
                "BM*BK must be a multiple of 4*256 to vectorize loads");
  static_assert((K23_BN * K23_BK) % (4 * K23_NUM_THREADS) == 0,
                "BN*BK must be a multiple of 4*256 to vectorize loads");

  dim3 gridDim(CEIL_DIV(N, K23_BN), CEIL_DIV(M, K23_BM));
  my_sgemmWarptiling<K23_BM, K23_BN, K23_BK, K23_WM, K23_WN, K23_WMITER, K23_WNITER, K23_TM,
                  K23_TN, K23_NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_my_SgemmWarptiling_compute(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  // Settings for A100
  // const uint K23_NUM_THREADS = 128;
  // const uint K23_BN = 128;
  // const uint K23_BM = 64;
  // const uint K23_BK = 16;
  // const uint K23_WN = 64;
  // const uint K23_WM = 32;
  // const uint K23_WNITER = 1;
  // const uint K23_TN = 4;
  // const uint K23_TM = 4;
  // Settings for A6000
  const uint K23_NUM_THREADS = 128;
  const uint K23_BN = 128;
  const uint K23_BM = 128;
  const uint K23_BK = 16;
  const uint K23_WN = 64;
  const uint K23_WM = 64;
  const uint K23_WNITER = 4;
  const uint K23_TN = 4;
  const uint K23_TM = 8;
  dim3 blockDim(K23_NUM_THREADS);

  constexpr uint NUM_WARPS = K23_NUM_THREADS / 32;

  // warptile in threadblocktile
  static_assert((K23_BN % K23_WN == 0) and (K23_BM % K23_WM == 0));
  static_assert((K23_BN / K23_WN) * (K23_BM / K23_WM) == NUM_WARPS);

  // threads in warpsubtile
  static_assert((K23_WM * K23_WN) % (WARPSIZE * K23_TM * K23_TN * K23_WNITER) ==
                0);
  constexpr uint K23_WMITER =
      (K23_WM * K23_WN) / (32 * K23_TM * K23_TN * K23_WNITER);
  // warpsubtile in warptile
  static_assert((K23_WM % K23_WMITER == 0) and (K23_WN % K23_WNITER == 0));

  static_assert((K23_NUM_THREADS * 4) % K23_BK == 0,
                "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of Bs during each iteraion)");
  static_assert((K23_NUM_THREADS * 4) % K23_BN == 0,
                "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of As during each iteration)");
  static_assert(K23_BN % (16 * K23_TN) == 0,
                "BN must be a multiple of 16*TN to avoid quantization effects");
  static_assert(K23_BM % (16 * K23_TM) == 0,
                "BM must be a multiple of 16*TM to avoid quantization effects");
  static_assert((K23_BM * K23_BK) % (4 * K23_NUM_THREADS) == 0,
                "BM*BK must be a multiple of 4*256 to vectorize loads");
  static_assert((K23_BN * K23_BK) % (4 * K23_NUM_THREADS) == 0,
                "BN*BK must be a multiple of 4*256 to vectorize loads");

  dim3 gridDim(CEIL_DIV(N, K23_BN), CEIL_DIV(M, K23_BM));
  my_sgemmWarptiling_compute<K23_BM, K23_BN, K23_BK, K23_WM, K23_WN, K23_WMITER, K23_WNITER, K23_TM,
                  K23_TN, K23_NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_my_SgemmWarptiling_write(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  // Settings for A100
  // const uint K23_NUM_THREADS = 128;
  // const uint K23_BN = 128;
  // const uint K23_BM = 64;
  // const uint K23_BK = 16;
  // const uint K23_WN = 64;
  // const uint K23_WM = 32;
  // const uint K23_WNITER = 1;
  // const uint K23_TN = 4;
  // const uint K23_TM = 4;
  // Settings for A6000
  const uint K23_NUM_THREADS = 128;
  const uint K23_BN = 128;
  const uint K23_BM = 128;
  const uint K23_BK = 16;
  const uint K23_WN = 64;
  const uint K23_WM = 64;
  const uint K23_WNITER = 4;
  const uint K23_TN = 4;
  const uint K23_TM = 8;
  dim3 blockDim(K23_NUM_THREADS);

  constexpr uint NUM_WARPS = K23_NUM_THREADS / 32;

  // warptile in threadblocktile
  static_assert((K23_BN % K23_WN == 0) and (K23_BM % K23_WM == 0));
  static_assert((K23_BN / K23_WN) * (K23_BM / K23_WM) == NUM_WARPS);

  // threads in warpsubtile
  static_assert((K23_WM * K23_WN) % (WARPSIZE * K23_TM * K23_TN * K23_WNITER) ==
                0);
  constexpr uint K23_WMITER =
      (K23_WM * K23_WN) / (32 * K23_TM * K23_TN * K23_WNITER);
  // warpsubtile in warptile
  static_assert((K23_WM % K23_WMITER == 0) and (K23_WN % K23_WNITER == 0));

  static_assert((K23_NUM_THREADS * 4) % K23_BK == 0,
                "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of Bs during each iteraion)");
  static_assert((K23_NUM_THREADS * 4) % K23_BN == 0,
                "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of As during each iteration)");
  static_assert(K23_BN % (16 * K23_TN) == 0,
                "BN must be a multiple of 16*TN to avoid quantization effects");
  static_assert(K23_BM % (16 * K23_TM) == 0,
                "BM must be a multiple of 16*TM to avoid quantization effects");
  static_assert((K23_BM * K23_BK) % (4 * K23_NUM_THREADS) == 0,
                "BM*BK must be a multiple of 4*256 to vectorize loads");
  static_assert((K23_BN * K23_BK) % (4 * K23_NUM_THREADS) == 0,
                "BN*BK must be a multiple of 4*256 to vectorize loads");

  dim3 gridDim(CEIL_DIV(N, K23_BN), CEIL_DIV(M, K23_BM));
  my_sgemmWarptiling_write<K23_BM, K23_BN, K23_BK, K23_WM, K23_WN, K23_WMITER, K23_WNITER, K23_TM,
                  K23_TN, K23_NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_my_SgemmWarptiling_pointer(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  // Settings for A100
  // const uint K23_NUM_THREADS = 128;
  // const uint K23_BN = 128;
  // const uint K23_BM = 64;
  // const uint K23_BK = 16;
  // const uint K23_WN = 64;
  // const uint K23_WM = 32;
  // const uint K23_WNITER = 1;
  // const uint K23_TN = 4;
  // const uint K23_TM = 4;
  // Settings for A6000
  const uint K23_NUM_THREADS = 128;
  const uint K23_BN = 128;
  const uint K23_BM = 128;
  const uint K23_BK = 16;
  const uint K23_WN = 64;
  const uint K23_WM = 64;
  const uint K23_WNITER = 4;
  const uint K23_TN = 4;
  const uint K23_TM = 8;
  dim3 blockDim(K23_NUM_THREADS);

  constexpr uint NUM_WARPS = K23_NUM_THREADS / 32;

  // warptile in threadblocktile
  static_assert((K23_BN % K23_WN == 0) and (K23_BM % K23_WM == 0));
  static_assert((K23_BN / K23_WN) * (K23_BM / K23_WM) == NUM_WARPS);

  // threads in warpsubtile
  static_assert((K23_WM * K23_WN) % (WARPSIZE * K23_TM * K23_TN * K23_WNITER) ==
                0);
  constexpr uint K23_WMITER =
      (K23_WM * K23_WN) / (32 * K23_TM * K23_TN * K23_WNITER);
  // warpsubtile in warptile
  static_assert((K23_WM % K23_WMITER == 0) and (K23_WN % K23_WNITER == 0));

  static_assert((K23_NUM_THREADS * 4) % K23_BK == 0,
                "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of Bs during each iteraion)");
  static_assert((K23_NUM_THREADS * 4) % K23_BN == 0,
                "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of As during each iteration)");
  static_assert(K23_BN % (16 * K23_TN) == 0,
                "BN must be a multiple of 16*TN to avoid quantization effects");
  static_assert(K23_BM % (16 * K23_TM) == 0,
                "BM must be a multiple of 16*TM to avoid quantization effects");
  static_assert((K23_BM * K23_BK) % (4 * K23_NUM_THREADS) == 0,
                "BM*BK must be a multiple of 4*256 to vectorize loads");
  static_assert((K23_BN * K23_BK) % (4 * K23_NUM_THREADS) == 0,
                "BN*BK must be a multiple of 4*256 to vectorize loads");

  dim3 gridDim(CEIL_DIV(N, K23_BN), CEIL_DIV(M, K23_BM));
  my_sgemmWarptiling_pointer<K23_BM, K23_BN, K23_BK, K23_WM, K23_WN, K23_WMITER, K23_WNITER, K23_TM,
                  K23_TN, K23_NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_my_SgemmWarptiling_pointer2(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  const uint K23_NUM_THREADS = 128;
  const uint K23_BN = 128;
  const uint K23_BM = 128;
  const uint K23_BK = 16;
  const uint K23_WN = 64;
  const uint K23_WM = 64;
  const uint K23_WNITER = 4;
  const uint K23_TN = 4;
  const uint K23_TM = 8;
  dim3 blockDim(K23_NUM_THREADS);

  constexpr uint NUM_WARPS = K23_NUM_THREADS / 32;

  // warptile in threadblocktile
  static_assert((K23_BN % K23_WN == 0) and (K23_BM % K23_WM == 0));
  static_assert((K23_BN / K23_WN) * (K23_BM / K23_WM) == NUM_WARPS);

  // threads in warpsubtile
  static_assert((K23_WM * K23_WN) % (WARPSIZE * K23_TM * K23_TN * K23_WNITER) ==
                0);
  constexpr uint K23_WMITER =
      (K23_WM * K23_WN) / (32 * K23_TM * K23_TN * K23_WNITER);
  // warpsubtile in warptile
  static_assert((K23_WM % K23_WMITER == 0) and (K23_WN % K23_WNITER == 0));

  static_assert((K23_NUM_THREADS * 4) % K23_BK == 0,
                "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of Bs during each iteraion)");
  static_assert((K23_NUM_THREADS * 4) % K23_BN == 0,
                "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of As during each iteration)");
  static_assert(K23_BN % (16 * K23_TN) == 0,
                "BN must be a multiple of 16*TN to avoid quantization effects");
  static_assert(K23_BM % (16 * K23_TM) == 0,
                "BM must be a multiple of 16*TM to avoid quantization effects");
  static_assert((K23_BM * K23_BK) % (4 * K23_NUM_THREADS) == 0,
                "BM*BK must be a multiple of 4*256 to vectorize loads");
  static_assert((K23_BN * K23_BK) % (4 * K23_NUM_THREADS) == 0,
                "BN*BK must be a multiple of 4*256 to vectorize loads");

  dim3 gridDim(CEIL_DIV(N, K23_BN), CEIL_DIV(M, K23_BM));
  my_sgemmWarptiling_pointer2<K23_BM, K23_BN, K23_BK, K23_WM, K23_WN, K23_WMITER, K23_WNITER, K23_TM,
                  K23_TN, K23_NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_my_SgemmWarptiling_device(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  const uint K23_NUM_THREADS = 128;
  const uint K23_BN = 128;
  const uint K23_BM = 128;
  const uint K23_BK = 16;
  const uint K23_WN = 64;
  const uint K23_WM = 64;
  const uint K23_WNITER = 4;
  const uint K23_TN = 4;
  const uint K23_TM = 8;
  dim3 blockDim(K23_NUM_THREADS);

  constexpr uint NUM_WARPS = K23_NUM_THREADS / 32;

  // warptile in threadblocktile
  static_assert((K23_BN % K23_WN == 0) and (K23_BM % K23_WM == 0));
  static_assert((K23_BN / K23_WN) * (K23_BM / K23_WM) == NUM_WARPS);

  // threads in warpsubtile
  static_assert((K23_WM * K23_WN) % (WARPSIZE * K23_TM * K23_TN * K23_WNITER) ==
                0);
  constexpr uint K23_WMITER =
      (K23_WM * K23_WN) / (32 * K23_TM * K23_TN * K23_WNITER);
  // warpsubtile in warptile
  static_assert((K23_WM % K23_WMITER == 0) and (K23_WN % K23_WNITER == 0));

  static_assert((K23_NUM_THREADS * 4) % K23_BK == 0,
                "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of Bs during each iteraion)");
  static_assert((K23_NUM_THREADS * 4) % K23_BN == 0,
                "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of As during each iteration)");
  static_assert(K23_BN % (16 * K23_TN) == 0,
                "BN must be a multiple of 16*TN to avoid quantization effects");
  static_assert(K23_BM % (16 * K23_TM) == 0,
                "BM must be a multiple of 16*TM to avoid quantization effects");
  static_assert((K23_BM * K23_BK) % (4 * K23_NUM_THREADS) == 0,
                "BM*BK must be a multiple of 4*256 to vectorize loads");
  static_assert((K23_BN * K23_BK) % (4 * K23_NUM_THREADS) == 0,
                "BN*BK must be a multiple of 4*256 to vectorize loads");

  dim3 gridDim(CEIL_DIV(N, K23_BN), CEIL_DIV(M, K23_BM));
  my_sgemmWarptiling_device<K23_BM, K23_BN, K23_BK, K23_WM, K23_WN, K23_WMITER, K23_WNITER, K23_TM,
                  K23_TN, K23_NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runSgemmDoubleBuffering(int M, int N, int K, float alpha, float *A,
                             float *B, float beta, float *C) {
  // Settings for A100
  // const uint K11_NUM_THREADS = 256;
  // const uint K11_BN = 128;
  // const uint K11_BM = 64;
  // const uint K11_BK = 16;
  // const uint K11_WN = 32;
  // const uint K11_WM = 32;
  // const uint K11_WNITER = 2;
  // const uint K11_TN = 4;
  // const uint K11_TM = 4;
  // Settings for A6000
  const uint K11_NUM_THREADS = 256;
  const uint K11_BN = 256;
  const uint K11_BM = 128;
  const uint K11_BK = 16;
  const uint K11_WN = 32;
  const uint K11_WM = 128;
  const uint K11_WNITER = 1;
  const uint K11_TN = 8;
  const uint K11_TM = 8;
  dim3 blockDim(K11_NUM_THREADS);

  constexpr uint NUM_WARPS = K11_NUM_THREADS / 32;

  // warptile in threadblocktile
  static_assert((K11_BN % K11_WN == 0) and (K11_BM % K11_WM == 0));
  static_assert((K11_BN / K11_WN) * (K11_BM / K11_WM) == NUM_WARPS);

  // threads in warpsubtile
  static_assert((K11_WM * K11_WN) % (WARPSIZE * K11_TM * K11_TN * K11_WNITER) ==
                0);
  constexpr uint K11_WMITER =
      (K11_WM * K11_WN) / (32 * K11_TM * K11_TN * K11_WNITER);
  // warpsubtile in warptile
  static_assert((K11_WM % K11_WMITER == 0) and (K11_WN % K11_WNITER == 0));

  static_assert((K11_NUM_THREADS / 2 * 4) % K11_BK == 0,
                "NUM_THREADS*4 must be multiple of BK to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of Bs during each iteraion)");
  static_assert((K11_NUM_THREADS / 2 * 4) % K11_BN == 0,
                "NUM_THREADS*4 must be multiple of BN to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of As during each iteration)");
  static_assert(K11_BN % (16 * K11_TN) == 0,
                "BN must be a multiple of 16*TN to avoid quantization effects");
  static_assert(K11_BM % (16 * K11_TM) == 0,
                "BM must be a multiple of 16*TM to avoid quantization effects");
  static_assert((K11_BM * K11_BK) % (4 * K11_NUM_THREADS / 2) == 0,
                "BM*BK must be a multiple of 4*256 to vectorize loads");
  static_assert((K11_BN * K11_BK) % (4 * K11_NUM_THREADS / 2) == 0,
                "BN*BK must be a multiple of 4*256 to vectorize loads");

  dim3 gridDim(CEIL_DIV(N, K11_BN), CEIL_DIV(M, K11_BM));
  sgemmDoubleBuffering<K11_BM, K11_BN, K11_BK, K11_WM, K11_WN, K11_WNITER,
                       K11_TM, K11_TN, K11_NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runSgemmDoubleBuffering2(int M, int N, int K, float alpha, float *A,
                              float *B, float beta, float *C) {
  // Settings for A6000
  const uint K12_NUM_THREADS = 128;
  const uint K12_BN = 128;
  const uint K12_BM = 128;
  const uint K12_BK = 16;
  const uint K12_WN = 64;
  const uint K12_WM = 64;
  const uint K12_WNITER = 4;
  const uint K12_TN = 4;
  const uint K12_TM = 8;
  dim3 blockDim(K12_NUM_THREADS);

  constexpr uint NUM_WARPS = K12_NUM_THREADS / 32;

  // warptile in threadblocktile
  static_assert((K12_BN % K12_WN == 0) and (K12_BM % K12_WM == 0));
  static_assert((K12_BN / K12_WN) * (K12_BM / K12_WM) == NUM_WARPS);

  // threads in warpsubtile
  static_assert((K12_WM * K12_WN) % (WARPSIZE * K12_TM * K12_TN * K12_WNITER) ==
                0);
  constexpr uint K12_WMITER =
      (K12_WM * K12_WN) / (32 * K12_TM * K12_TN * K12_WNITER);
  // warpsubtile in warptile
  static_assert((K12_WM % K12_WMITER == 0) and (K12_WN % K12_WNITER == 0));

  static_assert((K12_NUM_THREADS * 4) % K12_BK == 0,
                "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of Bs during each iteraion)");
  static_assert((K12_NUM_THREADS * 4) % K12_BN == 0,
                "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of As during each iteration)");
  static_assert(K12_BN % (16 * K12_TN) == 0,
                "BN must be a multiple of 16*TN to avoid quantization effects");
  static_assert(K12_BM % (16 * K12_TM) == 0,
                "BM must be a multiple of 16*TM to avoid quantization effects");
  static_assert((K12_BM * K12_BK) % (4 * K12_NUM_THREADS) == 0,
                "BM*BK must be a multiple of 4*256 to vectorize loads");
  static_assert((K12_BN * K12_BK) % (4 * K12_NUM_THREADS) == 0,
                "BN*BK must be a multiple of 4*256 to vectorize loads");

  dim3 gridDim(CEIL_DIV(N, K12_BN), CEIL_DIV(M, K12_BM));
  runSgemmDoubleBuffering2<K12_BM, K12_BN, K12_BK, K12_WM, K12_WN, K12_WNITER,
                           K12_TM, K12_TN, K12_NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_bf16_warptiling(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  const uint K23_NUM_THREADS = 128;
  const uint K23_BN = 128;
  const uint K23_BM = 128;
  const uint K23_BK = 16;
  const uint K23_WN = 64;
  const uint K23_WM = 64;
  const uint K23_WNITER = 4;
  const uint K23_TN = 4;
  const uint K23_TM = 8;
  dim3 blockDim(K23_NUM_THREADS);

  constexpr uint NUM_WARPS = K23_NUM_THREADS / 32;

  // warptile in threadblocktile
  static_assert((K23_BN % K23_WN == 0) and (K23_BM % K23_WM == 0));
  static_assert((K23_BN / K23_WN) * (K23_BM / K23_WM) == NUM_WARPS);

  // threads in warpsubtile
  static_assert((K23_WM * K23_WN) % (WARPSIZE * K23_TM * K23_TN * K23_WNITER) ==
                0);
  constexpr uint K23_WMITER =
      (K23_WM * K23_WN) / (32 * K23_TM * K23_TN * K23_WNITER);
  // warpsubtile in warptile
  static_assert((K23_WM % K23_WMITER == 0) and (K23_WN % K23_WNITER == 0));

  static_assert((K23_NUM_THREADS * 4) % K23_BK == 0,
                "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of Bs during each iteraion)");
  static_assert((K23_NUM_THREADS * 4) % K23_BN == 0,
                "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of As during each iteration)");
  static_assert(K23_BN % (16 * K23_TN) == 0,
                "BN must be a multiple of 16*TN to avoid quantization effects");
  static_assert(K23_BM % (16 * K23_TM) == 0,
                "BM must be a multiple of 16*TM to avoid quantization effects");
  static_assert((K23_BM * K23_BK) % (4 * K23_NUM_THREADS) == 0,
                "BM*BK must be a multiple of 4*256 to vectorize loads");
  static_assert((K23_BN * K23_BK) % (4 * K23_NUM_THREADS) == 0,
                "BN*BK must be a multiple of 4*256 to vectorize loads");

  dim3 gridDim(CEIL_DIV(N, K23_BN), CEIL_DIV(M, K23_BM));
  bf16_sgemmWarptiling<K23_BM, K23_BN, K23_BK, K23_WM, K23_WN, K23_WMITER, K23_WNITER, K23_TM,
                  K23_TN, K23_NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_bf16_tensorcore(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {

  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 128; 

  constexpr int WM = 32;
  constexpr int WN = 64;

  constexpr int TM = 16;
  constexpr int TN = 16;
  constexpr int TK = 16;
  
  constexpr int NUM_THREADS = 256;
  constexpr int WARPS_PER_BLOCK = NUM_THREADS / 32;

  constexpr int SHMEM_A_STRIDE = BK;
  const size_t shmem_size_for_A = BM * SHMEM_A_STRIDE * sizeof(bf16);

  
  constexpr int SHMEM_B_STRIDE = BN;
  const size_t shmem_size_for_B = BK * SHMEM_B_STRIDE * sizeof(bf16);
  
  
  const size_t shmem_size_for_CD = BM * BN * sizeof(float);
  
 
  const size_t required_shmem_for_AB = shmem_size_for_A + shmem_size_for_B;
  const size_t sharedMemSizeInBytes = std::max(required_shmem_for_AB, shmem_size_for_CD);

  int device;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  // if (sharedMemSizeInBytes > prop.sharedMemPerBlock) {
  //   printf("Requested shared memory size (%zu bytes) exceeds device limit (%zu bytes).\n",
  //          sharedMemSizeInBytes, prop.sharedMemPerBlock);
  //   return;
  // }

  // 告诉CUDA运行时为这个内核函数预留更多的共享内存
  // 这被称为 "opting in"
  cudaFuncSetAttribute(bf16_sgemm_tensorcore<BM, BN, BK, WM, WN, TM, TN, TK, NUM_THREADS, WARPS_PER_BLOCK>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       sharedMemSizeInBytes);
  
  dim3 blockDim(NUM_THREADS);

  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

  bf16_sgemm_tensorcore<BM, BN, BK, WM, WN, TM, TN, TK, NUM_THREADS, WARPS_PER_BLOCK>
      <<<gridDim, blockDim, sharedMemSizeInBytes>>>(
          M, N, K, alpha, A, B, beta, C);
}

void run_bf16_wmma_simple(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {

  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 64;

  constexpr int TM = 16;
  constexpr int TN = 16;
  constexpr int TK = 16;

  constexpr int NUM_THREADS = 128 * 4;

  //
  constexpr int SHMEM_A_STRIDE = BK;
  const size_t shmem_size_for_A = BM * SHMEM_A_STRIDE * sizeof(bf16);

  
  constexpr int SHMEM_B_STRIDE = BN;
  const size_t shmem_size_for_B = BK * SHMEM_B_STRIDE * sizeof(bf16);
  
  
  const size_t shmem_size_for_CD = BM * BN * sizeof(float);
  
 
  const size_t required_shmem_for_AB = shmem_size_for_A + shmem_size_for_B;
  const size_t sharedMemSizeInBytes = std::max(required_shmem_for_AB, shmem_size_for_CD);

  int device;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  if (sharedMemSizeInBytes > prop.sharedMemPerBlock) {
    printf("Requested shared memory size (%zu bytes) exceeds device limit (%zu bytes).\n",
           sharedMemSizeInBytes, prop.sharedMemPerBlock);
    return;
  }

  // 告诉CUDA运行时为这个内核函数预留更多的共享内存
  // 这被称为 "opting in"
  cudaFuncSetAttribute(bf16_wmma_simple<BM, BN, BK, TM, TN, TK, NUM_THREADS>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       sharedMemSizeInBytes);
  
  dim3 gridDim;
  dim3 blockDim;

  blockDim.x = 128;
  blockDim.y = 4;

  gridDim.x = (M + (TM * (blockDim.x / WARPSIZE) - 1)) / (TM * (blockDim.x / WARPSIZE));
  gridDim.y = (N + (TN * blockDim.y - 1)) / (TN * blockDim.y);

  bf16_wmma_simple<BM, BN, BK, TM, TN, TK, NUM_THREADS><<<gridDim, blockDim>>>(
          M, N, K, alpha, A, B, beta, C);
}

void run_bf16_wmma_test(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {

  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 64;

  constexpr int TM = 16;
  constexpr int TN = 16;
  constexpr int TK = 16;

  constexpr int NUM_THREADS = 128 * 4;
  
  dim3 gridDim;
  dim3 blockDim;

  blockDim.x = 128;
  blockDim.y = 4;

  gridDim.x = (M + (TM * (blockDim.x / WARPSIZE) - 1)) / (TM * (blockDim.x / WARPSIZE));
  gridDim.y = (N + (TN * blockDim.y - 1)) / (TN * blockDim.y);

  bf16_wmma_test<BM, BN, BK, TM, TN, TK, NUM_THREADS><<<gridDim, blockDim>>>(
          M, N, K, alpha, A, B, beta, C);
}

void run_bf16_wmma_cshare(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {

  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 64;

  constexpr int TM = 16;
  constexpr int TN = 16;
  constexpr int TK = 16;

  constexpr int NUM_THREADS = 128 * 4;
  
  dim3 gridDim;
  dim3 blockDim;

  blockDim.x = 128;
  blockDim.y = 4;

  gridDim.x = (M + (TM * (blockDim.x / WARPSIZE) - 1)) / (TM * (blockDim.x / WARPSIZE));
  gridDim.y = (N + (TN * blockDim.y - 1)) / (TN * blockDim.y);

  bf16_wmma_cshare<BM, BN, BK, TM, TN, TK, NUM_THREADS><<<gridDim, blockDim>>>(
          M, N, K, alpha, A, B, beta, C);
}

void run_bf16_wmma_reuse(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {

  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 64;

  constexpr int TM = 16;
  constexpr int TN = 16;
  constexpr int TK = 16;

  constexpr int NUM_THREADS = 128 * 4;
  constexpr int SHMEM_A_STRIDE = BK;
  const size_t shmem_size_for_A = BM * SHMEM_A_STRIDE * sizeof(bf16);

  
  constexpr int SHMEM_B_STRIDE = BN;
  const size_t shmem_size_for_B = BK * SHMEM_B_STRIDE * sizeof(bf16);
  
  
  const size_t shmem_size_for_CD = BM * BN * sizeof(float);
  
 
  const size_t required_shmem_for_AB = shmem_size_for_A + shmem_size_for_B;
  const size_t sharedMemSizeInBytes = std::max(required_shmem_for_AB, shmem_size_for_CD);

  int device;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  if (sharedMemSizeInBytes > prop.sharedMemPerBlock) {
    printf("Requested shared memory size (%zu bytes) exceeds device limit (%zu bytes).\n",
           sharedMemSizeInBytes, prop.sharedMemPerBlock);
    return;
  }

  // 告诉CUDA运行时为这个内核函数预留更多的共享内存
  // 这被称为 "opting in"
  cudaFuncSetAttribute(bf16_wmma_reuse<BM, BN, BK, TM, TN, TK, NUM_THREADS>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       sharedMemSizeInBytes);
  
  dim3 gridDim;
  dim3 blockDim;

  blockDim.x = 128;
  blockDim.y = 4;

  gridDim.x = (M + (TM * (blockDim.x / WARPSIZE) - 1)) / (TM * (blockDim.x / WARPSIZE));
  gridDim.y = (N + (TN * blockDim.y - 1)) / (TN * blockDim.y);

  bf16_wmma_reuse<BM, BN, BK, TM, TN, TK, NUM_THREADS><<<gridDim, blockDim, sharedMemSizeInBytes>>>(
          M, N, K, alpha, A, B, beta, C);
}

void run_bf16_wmma_warptiling(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {

  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 128;
  
  constexpr int WM = 32;
  constexpr int WN = 64;

  constexpr int TM = 16;
  constexpr int TN = 16;
  constexpr int TK = 16;

  constexpr int NUM_THREADS = 64 * 4;
  constexpr int SHMEM_A_STRIDE = BK;
  const size_t shmem_size_for_A = BM * SHMEM_A_STRIDE * sizeof(bf16);

  
  constexpr int SHMEM_B_STRIDE = BN;
  const size_t shmem_size_for_B = BK * SHMEM_B_STRIDE * sizeof(bf16);
  
  
  const size_t shmem_size_for_CD = BM * BN * sizeof(float);
  
 
  const size_t required_shmem_for_AB = shmem_size_for_A + shmem_size_for_B;
  const size_t sharedMemSizeInBytes = std::max(required_shmem_for_AB, shmem_size_for_CD);

  int device;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  // if (sharedMemSizeInBytes > prop.sharedMemPerBlock) {
  //   printf("Requested shared memory size (%zu bytes) exceeds device limit (%zu bytes).\n",
  //          sharedMemSizeInBytes, prop.sharedMemPerBlock);
  //   // return;
  // }

  // 告诉CUDA运行时为这个内核函数预留更多的共享内存
  // 这被称为 "opting in"
  cudaFuncSetAttribute(bf16_wmma_warptiling<BM, BN, BK, WM, WN, TM, TN, TK, NUM_THREADS>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       sharedMemSizeInBytes);
  
  dim3 gridDim;
  dim3 blockDim;

  blockDim.x = 64;
  blockDim.y = 4;

  gridDim.x = (M + BM - 1) / BM;
  gridDim.y = (N + BN - 1) / BN;

  bf16_wmma_warptiling<BM, BN, BK, WM, WN, TM, TN, TK, NUM_THREADS><<<gridDim, blockDim, sharedMemSizeInBytes>>>(
          M, N, K, alpha, A, B, beta, C);
}

void run_bf16_wmma_warptiling_float2(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {

  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 128;
  
  constexpr int WM = 32;
  constexpr int WN = 64;

  constexpr int TM = 16;
  constexpr int TN = 16;
  constexpr int TK = 16;

  constexpr int NUM_THREADS = 64 * 4;
  constexpr int SHMEM_A_STRIDE = BK;
  const size_t shmem_size_for_A = BM * SHMEM_A_STRIDE * sizeof(bf16);

  
  constexpr int SHMEM_B_STRIDE = BN;
  const size_t shmem_size_for_B = BK * SHMEM_B_STRIDE * sizeof(bf16);
  
  
  const size_t shmem_size_for_CD = BM * BN * sizeof(float);
  
 
  const size_t required_shmem_for_AB = shmem_size_for_A + shmem_size_for_B;
  const size_t sharedMemSizeInBytes = std::max(required_shmem_for_AB, shmem_size_for_CD);

  int device;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  // if (sharedMemSizeInBytes > prop.sharedMemPerBlock) {
  //   printf("Requested shared memory size (%zu bytes) exceeds device limit (%zu bytes).\n",
  //          sharedMemSizeInBytes, prop.sharedMemPerBlock);
  //   // return;
  // }

  // 告诉CUDA运行时为这个内核函数预留更多的共享内存
  // 这被称为 "opting in"
  cudaFuncSetAttribute(bf16_wmma_warptiling_float2<BM, BN, BK, WM, WN, TM, TN, TK, NUM_THREADS>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       sharedMemSizeInBytes);
  
  dim3 gridDim;
  dim3 blockDim;

  blockDim.x = 64;
  blockDim.y = 4;

  gridDim.x = (M + BM - 1) / BM;
  gridDim.y = (N + BN - 1) / BN;

  bf16_wmma_warptiling_float2<BM, BN, BK, WM, WN, TM, TN, TK, NUM_THREADS><<<gridDim, blockDim, sharedMemSizeInBytes>>>(
          M, N, K, alpha, A, B, beta, C);
}

void run_kernel(int kernel_num, int M, int N, int K, float alpha, float *A,
                float *B, float beta, float *C, cublasHandle_t handle) {
  switch (kernel_num) {
  case 0:
    runCublasFP32(handle, M, N, K, alpha, A, B, beta, C);
    break;
  case 1:
    run_sgemm_naive(M, N, K, alpha, A, B, beta, C);
    break;
  case 2:
    run_sgemm_coalesce(M, N, K, alpha, A, B, beta, C);
    break;
  case 14:
    run_my_sgemm_coalesce(M, N, K, alpha, A, B, beta, C);
    break;
  case 3:
    run_sgemm_shared_mem_block(M, N, K, alpha, A, B, beta, C);
    break;
  case 15:
    run_my_sgemm_shared_mem_block(M, N, K, alpha, A, B, beta, C);
    break;
  case 4:
    runSgemm1DBlocktiling(M, N, K, alpha, A, B, beta, C);
    break;
  case 16:
    run_my_Sgemm1DBlocktiling(M, N, K, alpha, A, B, beta, C);
    break;
  case 5:
    runSgemm2DBlocktiling(M, N, K, alpha, A, B, beta, C);
    break;
  case 17:
    run_my_Sgemm2DBlocktiling(M, N, K, alpha, A, B, beta, C);
    break;
  case 18:
    run_my_Sgemm2DBlocktiling_v2(M, N, K, alpha, A, B, beta, C);
    break;
  case 19:
    run_my_Sgemm2DBlocktiling_v3(M, N, K, alpha, A, B, beta, C);
    break;
  case 6:
    runSgemmVectorize(M, N, K, alpha, A, B, beta, C);
    break;
  case 20:
    run_my_SgemmVectorize(M, N, K, alpha, A, B, beta, C);
    break;
  case 7:
    runSgemmResolveBankConflicts(M, N, K, alpha, A, B, beta, C);
    break;
  case 22:
    runKernel20Autotuned(M, N, K, alpha, A, B, beta, C);
    break;
  case 21:
    run_my_ResolveBankConflicts(M, N, K, alpha, A, B, beta, C);
    break;
  case 8:
    runSgemmResolveBankExtraCol(M, N, K, alpha, A, B, beta, C);
    break;
  case 9:
    runSgemmAutotuned(M, N, K, alpha, A, B, beta, C);
    break;
  case 10:
    runSgemmWarptiling(M, N, K, alpha, A, B, beta, C);
    break;
  case 23:
    run_my_SgemmWarptiling(M, N, K, alpha, A, B, beta, C);
    break;
  case 24:
    run_my_SgemmWarptiling_compute(M, N, K, alpha, A, B, beta, C);
    break;
  case 25:
    run_my_SgemmWarptiling_write(M, N, K, alpha, A, B, beta, C);
    break;
  case 26:
    run_my_SgemmWarptiling_pointer(M, N, K, alpha, A, B, beta, C);
    break;
  case 27:
    run_my_SgemmWarptiling_pointer2(M, N, K, alpha, A, B, beta, C);
    break;
  case 28:
    run_my_SgemmWarptiling_device(M, N, K, alpha, A, B, beta, C);
    break;
  case 11:
    runSgemmDoubleBuffering(M, N, K, alpha, A, B, beta, C);
    break;
  case 12:
    runSgemmDoubleBuffering2(M, N, K, alpha, A, B, beta, C);
    break;
  case 13:
    run_my_sgemm_naive(M, N, K, alpha, A, B, beta, C);
    break;
  case 29:
    runCublasBF16(handle, M, N, K, alpha, A, B, beta, C);
    break;
  case 30:
    run_bf16_warptiling(M, N, K, alpha, A, B, beta, C);
    break;
  case 31:
    run_bf16_tensorcore(M, N, K, alpha, A, B, beta, C);
    break;
  case 32:
    run_bf16_wmma_simple(M, N, K, alpha, A, B, beta, C);
    break;
  case 33:
    run_bf16_wmma_test(M, N, K, alpha, A, B, beta, C);
    break;
  case 34:
    run_bf16_wmma_cshare(M, N, K, alpha, A, B, beta, C);
    break;
  case 35:
    run_bf16_wmma_reuse(M, N, K, alpha, A, B, beta, C);
    break;
  case 36:
    run_bf16_wmma_warptiling(M, N, K, alpha, A, B, beta, C);
    break;
  case 37:
    run_bf16_wmma_warptiling_float2(M, N, K, alpha, A, B, beta, C);
    break;
  default:
    throw std::invalid_argument("Unknown kernel number");
  }
}

void run_kernel(int kernel_num, int M, int N, int K, float alpha, 
                __nv_bfloat16 *A, __nv_bfloat16 *B, 
                float beta, float *C, cublasHandle_t handle) {
  switch (kernel_num) {
    case 38:
      runCublasBF16_AB(handle, M, N, K, alpha, A, B, beta, C);
      break;
    default:
      throw std::invalid_argument("Unknown BF16 kernel number or kernel does not support BF16 input.");
  }
}