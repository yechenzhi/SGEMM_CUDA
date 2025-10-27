#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*

Matrix sizes:
MxK * KxN = MxN

*/

__global__ void my_sgemm_global_mem_coalesce(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  const uint col = blockIdx.x * blockDim.x + threadIdx.x;
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < M && col < N){
    float tmp = 0.0;
    for( int i = 0; i < K; ++i ){
      tmp += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = alpha * tmp + beta * C[row * N + col];
  }
}