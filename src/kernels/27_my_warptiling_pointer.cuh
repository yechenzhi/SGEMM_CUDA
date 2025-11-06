#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
// const int WARPSIZE = 32; // warpSize is not constexpr

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS) 
    my_sgemmWarptiling_pointer2(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  const int row_start_block = blockIdx.y * BM;
  const int col_start_block = blockIdx.x * BN;
  
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];
  float threadResults[WMITER*TM*WNITER*TN] = {0.0};
  float regM[WMITER * TM];
  float regN[WNITER * TN];

  const int tid = threadIdx.x;

  // used for loading A and B to shared memory
  const int as_row = tid / (BK / 4);
  const int as_col = tid % (BK / 4);
  constexpr int stride_As = NUM_THREADS / (BK / 4);
  const int bs_row = tid / (BN / 4);
  const int bs_col = tid % (BN / 4);
  constexpr int stride_Bs = NUM_THREADS / (BN / 4);

  //used for computing within the warptile
  //warp in block
  const int warpIdx = tid / WARPSIZE; 
  const int warpCol = warpIdx % (BN / WN);
  const int warpRow = warpIdx / (BN / WN);
  //warp shape
  constexpr int WNSUB = WN / WNITER;
  constexpr int WMSUB = WM / WMITER;
  //thread in warp
  const int tid_in_warp = tid % WARPSIZE;
  const int col_in_warp = tid_in_warp % (WNSUB / TN);
  const int row_in_warp = tid_in_warp / (WNSUB / TN);

  A += row_start_block * K;
  B += col_start_block;
  C += (row_start_block + warpRow * WM) * N + col_start_block + warpCol * WN;


  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    for(int row_offset = 0; row_offset < BM; row_offset += stride_As) {
        float4 tmp =
            reinterpret_cast<const float4 *>(&A[(row_offset + as_row) * K + as_col * 4])[0];
        As[(as_col * 4 + 0) * BM + row_offset + as_row] = tmp.x; 
        As[(as_col * 4 + 1) * BM + row_offset + as_row] = tmp.y;
        As[(as_col * 4 + 2) * BM + row_offset + as_row] = tmp.z;
        As[(as_col * 4 + 3) * BM + row_offset + as_row] = tmp.w;
    }
    for(int row_offset = 0; row_offset < BK; row_offset += stride_Bs) {
        reinterpret_cast<float4 *>(&Bs[(row_offset + bs_row) * BN + bs_col * 4])[0] =
            reinterpret_cast<const float4 *>(&B[(row_offset + bs_row) * N + bs_col * 4])[0];
    }
    __syncthreads();

    for(int k = 0; k < BK; ++k) {
            for(int wm_iter = 0; wm_iter < WMITER; ++wm_iter) {
                for(int tmIdx = 0; tmIdx < TM; ++tmIdx) {
                    regM[wm_iter * TM + tmIdx] =
                        As[k * BM + warpRow * WM + wm_iter * WMSUB + row_in_warp * TM + tmIdx];
                }
            }
            for(int wn_iter = 0; wn_iter < WNITER; ++wn_iter) {
                for(int tnIdx = 0; tnIdx < TN; ++tnIdx) {
                    regN[wn_iter * TN + tnIdx] =
                        Bs[k * BN + warpCol * WN + wn_iter * WNSUB + col_in_warp * TN + tnIdx];
                }
            }
            for(int wm_iter = 0; wm_iter < WMITER; ++wm_iter) {
                for(int wn_iter = 0; wn_iter < WNITER; ++wn_iter) {
                    for (int tmIdx = 0; tmIdx < TM; ++tmIdx) {
                        for (int tnIdx = 0; tnIdx < TN; ++tnIdx) {
                            threadResults[(wm_iter * TM + tmIdx) * WNITER * TN + wn_iter * TN + tnIdx] += 
                                regM[wm_iter * TM + tmIdx] *
                                regN[wn_iter * TN + tnIdx];
                        }
                    }
                }
            }
        }
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down
    __syncthreads();
   
  }

  for(int wm_iter = 0; wm_iter < WMITER; ++wm_iter) {
      for(int wn_iter = 0; wn_iter < WNITER; ++wn_iter) {
          float *C_interim = C + (wm_iter * WMSUB) * N + (wn_iter * WNSUB);
        //   float *C_interim = C + (wm_iter * WMSUB + row_in_warp * TM) * N + wn_iter * WNSUB + col_in_warp * TN;
          for (int tmIdx = 0; tmIdx < TM; ++tmIdx) {
              for (int tnIdx = 0; tnIdx < TN; tnIdx += 4) {

                //   float4 tmp = reinterpret_cast<float4 *>(&C_interim[tmIdx * N + tnIdx])[0];
                  float4 tmp = reinterpret_cast<float4 *>(
                    &C_interim[(row_in_warp * TM + tmIdx) * N + (col_in_warp * TN + tnIdx)])[0];
                  const int i = (wm_iter * TM + tmIdx) * (WNITER * TN) + wn_iter * TN + tnIdx;

                  tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
                  tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
                  tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
                  tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
                  reinterpret_cast<float4 *>(
                    &C_interim[(row_in_warp * TM + tmIdx) * N + (col_in_warp * TN + tnIdx)])[0] = tmp;
              }
          }
      }
  }
   
}
