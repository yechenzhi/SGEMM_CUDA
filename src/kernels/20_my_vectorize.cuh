#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void my_sgemmVectorize(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  const int row_start = blockIdx.y * BM + threadIdx.y * TM;
  const int col_start = blockIdx.x * BN + threadIdx.x * TN;
  
  __shared__ float As[BK][BM];
  __shared__ float Bs[BK][BN];

  float threadResults[TM][TN] = {{0.0}};
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
//   const int strideA = (blockDim.x * blockDim.y) / BK;
//   const int strideB = (blockDim.x * blockDim.y) / BN;

  const int as_row = tid / (BK / 4);
  const int as_col = tid % (BK / 4);
  const int bs_row = tid / (BN / 4);
  const int bs_col = tid % (BN / 4);

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // for A: row = blockIdx.y * blockDim.y + threadIdx.x / BK, col = bkIdx + threadIdx.x % BK
    // for B: row = bkIdx + threadIdx.y, col = col
    // float4 tmp =
    //     reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
    // As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    // As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    // As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    // As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;
    float4 tmp =
        reinterpret_cast<const float4 *>(&A[(blockIdx.y * BM + as_row) * K + bkIdx + as_col * 4])[0];
    As[as_col * 4 + 0][as_row] = tmp.x;
    As[as_col * 4 + 1][as_row] = tmp.y;
    As[as_col * 4 + 2][as_row] = tmp.z;
    As[as_col * 4 + 3][as_row] = tmp.w;

    reinterpret_cast<float4 *>(&Bs[bs_row][bs_col * 4])[0] =
        reinterpret_cast<const float4 *>(&B[(bkIdx + bs_row) * N + blockIdx.x * BN + bs_col * 4])[0];
    __syncthreads();

    for(int k = 0; k < BK; ++k) {
        for (int tmIdx = 0; tmIdx < TM; ++tmIdx) {
            for (int tnIdx = 0; tnIdx < TN; ++tnIdx) {
                threadResults[tmIdx][tnIdx] += As[k][threadIdx.y  * TM + tmIdx] *
                                               Bs[k][threadIdx.x * TN + tnIdx];
            }
        }
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }

//   for (int tmIdx = 0; tmIdx < TM; ++tmIdx) {
//         for (int tnIdx = 0; tnIdx < TN; ++tnIdx) {
//             C[(row_start + tmIdx) * N + (col_start + tnIdx)] =
//             alpha * threadResults[tmIdx][tnIdx] + beta * C[(row_start + tmIdx) * N + (col_start + tnIdx)];  
//         }
//     } 
  for (int tmIdx = 0; tmIdx < TM; ++tmIdx) {
        for (int tnIdx = 0; tnIdx < TN; tnIdx += 4) {
            float4 tmp = reinterpret_cast<float4 *>(
                &C[(row_start + tmIdx) * N + (col_start + tnIdx)])[0];
            tmp.x = alpha * threadResults[tmIdx][tnIdx + 0] + beta * tmp.x;
            tmp.y = alpha * threadResults[tmIdx][tnIdx + 1] + beta * tmp.y;
            tmp.z = alpha * threadResults[tmIdx][tnIdx + 2] + beta * tmp.z;
            tmp.w = alpha * threadResults[tmIdx][tnIdx + 3] + beta * tmp.w;
            reinterpret_cast<float4 *>(
                &C[(row_start + tmIdx) * N + (col_start + tnIdx)])[0] = tmp;   
        }
    }
}