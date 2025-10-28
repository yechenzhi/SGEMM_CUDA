#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM>
__global__ void my_sgemm1DBlocktiling(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  const int row_start = blockIdx.y * BM + threadIdx.y * TM;
  const int col = blockIdx.x * BN + threadIdx.x;
  
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  float threadResults[TM] = {0.0};
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    const int as_load_row = tid / BK; // 当前线程负责加载As的哪一行
    const int as_load_col = tid % BK; // 当前线程负责加载As的哪一列
    
    // 计算对应的全局A矩阵地址
    int gmem_A_row = blockIdx.y * BM + as_load_row;
    int gmem_A_col = bkIdx + as_load_col;
    
    // 使用【正确的】地址进行边界检查和加载
    if (gmem_A_row < M && gmem_A_col < K) {
        As[as_load_row][as_load_col] = A[gmem_A_row * K + gmem_A_col];
    } else {
        As[as_load_row][as_load_col] = 0.0f;
    }

    const int bs_load_row = tid / BN; // 当前线程负责加载Bs的哪一行
    const int bs_load_col = tid % BN; // 当前线程负责加载Bs的哪一列
    
    // 计算对应的全局A矩阵地址
    int gmem_B_row = bkIdx + bs_load_row;
    int gmem_B_col = blockIdx.x * BN + bs_load_col; 
    
    // 使用【正确的】地址进行边界检查和加载
    if (gmem_B_row < K && gmem_B_col < N) {
        Bs[bs_load_row][bs_load_col] = B[gmem_B_row * N + gmem_B_col];
    } else {
        Bs[bs_load_row][bs_load_col] = 0.0f;
    }

    // block threads in this block until cache is fully populated
    __syncthreads();

    // execute the dotproduct on the currently cached block
    for (int tmIdx = 0; tmIdx < TM; ++tmIdx) {
        for (int k = 0; k < BK; ++k) {
            threadResults[tmIdx] += As[threadIdx.y  * TM + tmIdx][k] *
                                   Bs[k][threadIdx.x];
        }
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }

  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(row_start + resIdx) * N + col] =
        alpha * threadResults[resIdx] + beta * C[(row_start + resIdx) * N + col];
  }
}