#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
// const int WARPSIZE = 32; // warpSize is not constexpr

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS) 
    my_sgemmWarptiling_write(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  const int row_start_block = blockIdx.y * BM;
  const int col_start_block = blockIdx.x * BN;
  
  __shared__ float As[BK][BM];
  __shared__ float Bs[BK][BN];
  float threadResults[WMITER*TM][WNITER*TN] = {{0.0}};
  float regM[WMITER * TM] = {0.0};
  float regN[WNITER * TN] = {0.0};

  const int tid = threadIdx.x;

  // used for loading A and B to shared memory
  const int as_row = tid / (BK / 4);
  const int as_col = tid % (BK / 4);
  const int stride_As = NUM_THREADS / (BK / 4);
  const int bs_row = tid / (BN / 4);
  const int bs_col = tid % (BN / 4);
  const int stride_Bs = NUM_THREADS / (BN / 4);

  //used for computing within the warptile
  //warp in block
  const int warpIdx = tid / WARPSIZE; 
  const int warpCol = warpIdx % (BN / WN);
  const int warpRow = warpIdx / (BN / WN);
  //warp shape
  const int WNSUB = WN / WNITER;
  const int WMSUB = WM / WMITER;
  //thread in warp
  const int tid_in_warp = tid % WARPSIZE;
  const int col_in_warp = tid_in_warp % (WNSUB / TN);
  const int row_in_warp = tid_in_warp / (WNSUB / TN);


  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    for(int row_offset = 0; row_offset < BM; row_offset += stride_As) {
        float4 tmp =
            reinterpret_cast<const float4 *>(&A[(row_start_block + row_offset + as_row) * K + bkIdx + as_col * 4])[0];
        As[as_col * 4 + 0][row_offset + as_row] = tmp.x; 
        As[as_col * 4 + 1][row_offset + as_row] = tmp.y;
        As[as_col * 4 + 2][row_offset + as_row] = tmp.z;
        As[as_col * 4 + 3][row_offset + as_row] = tmp.w;
    }
    for(int row_offset = 0; row_offset < BK; row_offset += stride_Bs) {
        reinterpret_cast<float4 *>(&Bs[row_offset + bs_row][bs_col * 4])[0] =
            reinterpret_cast<const float4 *>(&B[(bkIdx + row_offset + bs_row) * N + col_start_block + bs_col * 4])[0];
    }
    __syncthreads();

    for(int k = 0; k < BK; ++k) {
        for(int wm_iter = 0; wm_iter < WMITER; ++wm_iter) {
            for(int tmIdx = 0; tmIdx < TM; ++tmIdx) {
                regM[wm_iter * TM + tmIdx] =
                    As[k][warpRow * WM + wm_iter * WMSUB + row_in_warp * TM + tmIdx];
            }
        }
        for(int wn_iter = 0; wn_iter < WNITER; ++wn_iter) {
            for(int tnIdx = 0; tnIdx < TN; ++tnIdx) {
                regN[wn_iter * TN + tnIdx] =
                    Bs[k][warpCol * WN + wn_iter * WNSUB + col_in_warp * TN + tnIdx];
            }
        }
        for(int wm_iter = 0; wm_iter < WMITER; ++wm_iter) {
            for(int wn_iter = 0; wn_iter < WNITER; ++wn_iter) {
                for (int tmIdx = 0; tmIdx < TM; ++tmIdx) {
                    for (int tnIdx = 0; tnIdx < TN; ++tnIdx) {
                        threadResults[wm_iter * TM + tmIdx][wn_iter * TN + tnIdx] += 
                            regM[wm_iter * TM + tmIdx] *
                            regN[wn_iter * TN + tnIdx];
                    }
                }
            }
        }
    }
    __syncthreads();
   
  }

  const int warp_start_row = row_start_block + warpRow * WM;
  const int warp_start_col = col_start_block + warpCol * WN;
  for(int wm_iter = 0; wm_iter < WMITER; ++wm_iter) {
      for(int wn_iter = 0; wn_iter < WNITER; ++wn_iter) {
          const int tid_start_row = warp_start_row + wm_iter * WMSUB + row_in_warp * TM;
          const int tid_start_col = warp_start_col + wn_iter * WNSUB + col_in_warp * TN;
          for (int tmIdx = 0; tmIdx < TM; ++tmIdx) {
              for (int tnIdx = 0; tnIdx < TN; tnIdx += 4) {
                  float4 tmp = reinterpret_cast<float4 *>(&C[ (tid_start_row + tmIdx) * N + tid_start_col + tnIdx])[0];
                  tmp.x = alpha * threadResults[wm_iter * TM + tmIdx][wn_iter * TN + tnIdx + 0] + beta * tmp.x;
                  tmp.y = alpha * threadResults[wm_iter * TM + tmIdx][wn_iter * TN + tnIdx + 1] + beta * tmp.y;
                  tmp.z = alpha * threadResults[wm_iter * TM + tmIdx][wn_iter * TN + tnIdx + 2] + beta * tmp.z;
                  tmp.w = alpha * threadResults[wm_iter * TM + tmIdx][wn_iter * TN + tnIdx + 3] + beta * tmp.w;
                  reinterpret_cast<float4 *>(
                      &C[(tid_start_row + tmIdx) * N + (tid_start_col + tnIdx)])[0] = tmp;
              }
          }
      }
  }
   
}
