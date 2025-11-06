#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h> 

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
using bf16 = __nv_bfloat16;
// const int WARPSIZE = 32; // warpSize is not constexpr

namespace bf16wt {
    template <const int BM, const int BN, const int BK, const int stride_As, const int stride_Bs>
    __device__ void loadFromGmem(int N, int K, const float *A, const float *B,
                                 bf16 *As, bf16 *Bs,
                                 int as_row, int as_col, int bs_row, int bs_col) {

        for(uint row_offset = 0; row_offset < BM; row_offset += stride_As) {
            float4 tmp_fp32 =
                reinterpret_cast<const float4 *>(&A[(row_offset + as_row) * K + as_col * 4])[0];
            bf16 tmp_bf16_x = __float2bfloat16(tmp_fp32.x);
            bf16 tmp_bf16_y = __float2bfloat16(tmp_fp32.y);
            bf16 tmp_bf16_z = __float2bfloat16(tmp_fp32.z);
            bf16 tmp_bf16_w = __float2bfloat16(tmp_fp32.w);

            As[(as_col * 4 + 0) * BM + row_offset + as_row] = tmp_bf16_x; 
            As[(as_col * 4 + 1) * BM + row_offset + as_row] = tmp_bf16_y;
            As[(as_col * 4 + 2) * BM + row_offset + as_row] = tmp_bf16_z;
            As[(as_col * 4 + 3) * BM + row_offset + as_row] = tmp_bf16_w;
        }
        for(uint row_offset = 0; row_offset < BK; row_offset += stride_Bs) {
            float4 tmp_fp32 = 
                reinterpret_cast<const float4 *>(&B[(row_offset + bs_row) * N + bs_col * 4])[0];
            bf16* target_addr = &Bs[(row_offset + bs_row) * BN + bs_col * 4];
            target_addr[0] = __float2bfloat16(tmp_fp32.x);
            target_addr[1] = __float2bfloat16(tmp_fp32.y);
            target_addr[2] = __float2bfloat16(tmp_fp32.z);
            target_addr[3] = __float2bfloat16(tmp_fp32.w);
        }
    }

    template <const int BM, const int BK, const int BN, const int WM, const int WN, const int WMITER, const int WNITER,
              const int TM, const int TN, const int WMSUB, const int WNSUB>
    __device__ void processFromSmem(bf16 *regM, bf16 *regN, float *threadResults, const bf16 *As,
                    const bf16 *Bs, const uint warpRow, const uint warpCol,
                    const uint row_in_warp, const uint col_in_warp) {
        for(uint k = 0; k < BK; ++k) {
            for(uint wm_iter = 0; wm_iter < WMITER; ++wm_iter) {
                for(uint tmIdx = 0; tmIdx < TM; ++tmIdx) {
                    regM[wm_iter * TM + tmIdx] =
                        As[k * BM + warpRow * WM + wm_iter * WMSUB + row_in_warp * TM + tmIdx];
                }
            }
            for(uint wn_iter = 0; wn_iter < WNITER; ++wn_iter) {
                for(uint tnIdx = 0; tnIdx < TN; ++tnIdx) {
                    regN[wn_iter * TN + tnIdx] =
                        Bs[k * BN + warpCol * WN + wn_iter * WNSUB + col_in_warp * TN + tnIdx];
                }
            }
            for(uint wm_iter = 0; wm_iter < WMITER; ++wm_iter) {
                for(uint wn_iter = 0; wn_iter < WNITER; ++wn_iter) {
                    for (uint tmIdx = 0; tmIdx < TM; ++tmIdx) {
                        for (uint tnIdx = 0; tnIdx < TN; ++tnIdx) {
                            threadResults[(wm_iter * TM + tmIdx) * WNITER * TN + wn_iter * TN + tnIdx] += 
                                __bfloat162float(regM[wm_iter * TM + tmIdx]) *
                                __bfloat162float(regN[wn_iter * TN + tnIdx]);
                        }
                    }
                }
            }
        }
    }
} // namespace bf16wt

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS) 
    bf16_sgemmWarptiling(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  const uint row_start_block = blockIdx.y * BM;
  const uint col_start_block = blockIdx.x * BN;
  
  __shared__ bf16 As[BM * BK];
  __shared__ bf16 Bs[BK * BN];
  float threadResults[WMITER*TM*WNITER*TN] = {0.0};
  bf16 regM_inner[WMITER * TM];
  bf16 regN_inner[WNITER * TN];

  const uint tid = threadIdx.x;

  // used for loading A and B to shared memory
  const uint as_row = tid / (BK / 4);
  const uint as_col = tid % (BK / 4);
  constexpr uint stride_As = NUM_THREADS / (BK / 4);
  const uint bs_row = tid / (BN / 4);
  const uint bs_col = tid % (BN / 4);
  constexpr uint stride_Bs = NUM_THREADS / (BN / 4);

  //used for computing within the warptile
  //warp in block
  const uint warpIdx = tid / WARPSIZE; 
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);
  //warp shape
  constexpr uint WNSUB = WN / WNITER;
  constexpr uint WMSUB = WM / WMITER;
  //thread in warp
  const uint tid_in_warp = tid % WARPSIZE;
  const uint col_in_warp = tid_in_warp % (WNSUB / TN);
  const uint row_in_warp = tid_in_warp / (WNSUB / TN);

  A += row_start_block * K;
  B += col_start_block;
  C += (row_start_block + warpRow * WM) * N + col_start_block + warpCol * WN;


  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    bf16wt::loadFromGmem<BM, BN, BK, stride_As, stride_Bs>(N, K, A, B, 
                    As, Bs, as_row, as_col, bs_row, bs_col);
    __syncthreads();

    bf16wt::processFromSmem<BM, BK, BN, WM, WN, WMITER, WNITER, TM, TN, WMSUB, WNSUB>(
        regM_inner, regN_inner, threadResults, 
        As, Bs, warpRow, warpCol, row_in_warp, col_in_warp);
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down
    __syncthreads();
   
  }

  for(uint wm_iter = 0; wm_iter < WMITER; ++wm_iter) {
      for(uint wn_iter = 0; wn_iter < WNITER; ++wn_iter) {
          float *C_interim = C + (wm_iter * WMSUB) * N + (wn_iter * WNSUB);
        //   float *C_interim = C + (wm_iter * WMSUB + row_in_warp * TM) * N + wn_iter * WNSUB + col_in_warp * TN;
          for (uint tmIdx = 0; tmIdx < TM; ++tmIdx) {
              for (uint tnIdx = 0; tnIdx < TN; tnIdx += 4) {

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
