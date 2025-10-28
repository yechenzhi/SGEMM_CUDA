#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BN / TN) * (BM / TM))
    my_sgemm2DBlocktiling_v2(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  const int row_start = blockIdx.y * BM + threadIdx.y * TM;
  const int col_start = blockIdx.x * BN + threadIdx.x * TN;
  
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  float threadResults[TM][TN] = {{0.0}};
  // register caches for As and Bs
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};
  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // for A: row = blockIdx.y * blockDim.y + threadIdx.x / BK, col = bkIdx + threadIdx.x % BK
    // for B: row = bkIdx + threadIdx.y, col = col
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int strideA = BM * BK / (blockDim.x * blockDim.y);
    const int strideB = BK * BN / (blockDim.x * blockDim.y);

    const int as_row_start = tid / BK * strideA;
    const int as_col = tid % BK;
    const int bs_row_start = tid / BN * strideB;
    const int bs_col = tid % BN;

    for(int row_offset = 0; row_offset < strideA; ++row_offset) {
        int as_row = as_row_start + row_offset;
        int Arow = blockIdx.y * BM + as_row;
        int Acol = bkIdx + as_col;
        if (Arow < M && Acol < K) {
            As[as_row][as_col] = A[Arow * K + Acol];
        } else {
            As[as_row][as_col] = 0.0;
        }
    }
    for(int row_offset = 0; row_offset < strideB; ++row_offset) {
        int bs_row = bs_row_start + row_offset;
        int Brow = bkIdx + bs_row;
        int Bcol = blockIdx.x * BN + bs_col;
        if (Brow < K && Bcol < N) {
            Bs[bs_row][bs_col] = B[Brow * N + Bcol];
        } else {
            Bs[bs_row][bs_col] = 0.0;
        }
    }
    // block threads in this block until cache is fully populated
    __syncthreads();
    for(uint k = 0; k < BK; ++k){
        for(uint i = 0; i < TM; ++i){
            regM[i] = As[threadIdx.y * TM + i][k];
        }
        for(uint j = 0; j < TN; ++j){
            regN[j] = Bs[k][threadIdx.x * TN + j];
        }
        for(uint i = 0; i < TM; ++i){
            for(uint j = 0; j < TN; ++j){
                threadResults[i][j] += regM[i] * regN[j];
            }
        }
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }

  for (int tmIdx = 0; tmIdx < TM; ++tmIdx) {
        for (int tnIdx = 0; tnIdx < TN; ++tnIdx) {
            C[(row_start + tmIdx) * N + (col_start + tnIdx)] =
            alpha * threadResults[tmIdx][tnIdx] + beta * C[(row_start + tmIdx) * N + (col_start + tnIdx)];  
        }
    } 
}