#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h> 
#include <mma.h> 

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
using bf16 = __nv_bfloat16;
using namespace nvcuda;
// const int WARPSIZE = 32; // warpSize is not constexpr

namespace mywt {
    template <const int BM, const int BN, const int BK, const int stride_As, const int stride_Bs>
    __device__ void loadFromGmem(int N, int K, const float *A, const float *B,
                                 bf16 *As, bf16 *Bs,
                                 int as_row, int as_col, int bs_row, int bs_col) {

        for(uint row_offset = 0; row_offset < BM; row_offset += stride_As) {
            float4 tmp_fp32 =
                reinterpret_cast<const float4 *>(&A[(row_offset + as_row) * K + as_col * 4])[0];
            bf16* target_addr = &As[(row_offset + as_row) * BK + as_col * 4];
            target_addr[0] = __float2bfloat16(tmp_fp32.x);
            target_addr[1] = __float2bfloat16(tmp_fp32.y);
            target_addr[2] = __float2bfloat16(tmp_fp32.z);
            target_addr[3] = __float2bfloat16(tmp_fp32.w);
        }
        for(uint row_offset = 0; row_offset < BK; row_offset += stride_Bs) {
            float4 tmp_fp32 = 
                reinterpret_cast<const float4 *>(&B[(row_offset + bs_row) * N + bs_col * 4])[0];
            bf16* target_addr = &Bs[(row_offset + bs_row) * BN + bs_col * 4];
            bf16 tmp_bf16_x = __float2bfloat16(tmp_fp32.x);
            bf16 tmp_bf16_y = __float2bfloat16(tmp_fp32.y);
            bf16 tmp_bf16_z = __float2bfloat16(tmp_fp32.z);
            bf16 tmp_bf16_w = __float2bfloat16(tmp_fp32.w);
            Bs[(bs_col * 4 + 0) * BK + row_offset + bs_row] = tmp_bf16_x;
            Bs[(bs_col * 4 + 1) * BK + row_offset + bs_row] = tmp_bf16_y;
            Bs[(bs_col * 4 + 2) * BK + row_offset + bs_row] = tmp_bf16_z;
            Bs[(bs_col * 4 + 3) * BK + row_offset + bs_row] = tmp_bf16_w;
        }
    }
} // namespace mywt

/*
 * @tparam BM The threadblock size for M dimension SMEM caching.
 * @tparam BN The threadblock size for N dimension SMEM caching.
 * @tparam BK The threadblock size for K dimension SMEM caching.
 * @tparam CHUNK_K The number of K dimension chunks processed by each threadblock.
 */

template <const int BM, const int BN, const int BK,
          const int WM, const int WN, 
          const int TM, const int TN, const int TK, 
          const int NUM_THREADS, const int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(NUM_THREADS) 
    bf16_sgemm_tensorcore(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) { 

  const uint WARPS_PER_BLOCK_ROW = BN / WN;
  const uint WARPS_PER_BLOCK_COL = BM / WM;
  const uint TILES_PER_WARP_ROW = WN / TN;
  const uint TILES_PER_WARP_COL = WM / TM;

  const uint tid = threadIdx.x;
  const uint warpId = tid / 32;
  const uint laneId = tid % 32;


  const uint global_block_row = blockIdx.y * BM;
  const uint global_block_col = blockIdx.x * BN;

  const uint warpRow = warpId / WARPS_PER_BLOCK_ROW;
  const uint warpCol = warpId % WARPS_PER_BLOCK_ROW;

  extern __shared__ char shmem[];
  bf16* shmem_A = (bf16*)shmem;
  bf16* shmem_B = shmem_A + BM * BK;
  float* shmem_C = (float*)shmem;

  // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
  // each tile computation. Technically this is not generally correct (may result
  // in a loss of precision). Zero still needs to be specially handled though.
  beta /= alpha;

  // copy C from global memory to shared memory
  // each thread loads float4 (BM * BN / NUM_THREADS / 4) times
  // we simply use a warp copy a row (32 * 4) at a time
  // and copy 16 times (BM / WARPS_PER_BLOCK)
  // here we assume BN = 128 = 32 * 4
  const uint shmem_C_row = warpId;
  const uint shmem_C_col = laneId * 4;
  for(uint i = 0; i < BM / WARPS_PER_BLOCK; ++i) {
      float4 tmp = reinterpret_cast<const float4*>(
          &C[(global_block_row + shmem_C_row + i * WARPS_PER_BLOCK) * N + global_block_col + shmem_C_col])[0];
      reinterpret_cast<float4*>(
          &shmem_C[(shmem_C_row + i * WARPS_PER_BLOCK) * BN + shmem_C_col])[0] = tmp;
  }
  __syncthreads();

  // copy Cs from shared memory to registers
  wmma::fragment<wmma::accumulator, TM, TN, TK, float> c_frag[TILES_PER_WARP_COL][TILES_PER_WARP_ROW];
  for(uint i = 0; i < TILES_PER_WARP_COL; ++i) {
      for(uint j = 0; j < TILES_PER_WARP_ROW; ++j) {
          wmma::load_matrix_sync(c_frag[i][j],
              &shmem_C[(warpRow * WM + i * TM) * BN + warpCol * WN + j * TN],
              BN, wmma::mem_row_major);
      }
  }
  __syncthreads();

  // scale C matrix
  for(uint i = 0; i < TILES_PER_WARP_COL; ++i) {
      for(uint j = 0; j < TILES_PER_WARP_ROW; ++j) {
        for(uint t = 0; t < c_frag[i][j].num_elements; ++t) {
            c_frag[i][j].x[t] *= beta;
        }
      }
  }

  // main loop
  for(uint k_base = 0; k_base < K; k_base += BK) {
      // copy A and B from global memory to shared memory
      const uint as_row = tid / (BK / 4);
      const uint as_col = tid % (BK / 4);
      constexpr uint stride_As = NUM_THREADS / (BK / 4);
      const uint bs_row = tid / (BN / 4);
      const uint bs_col = tid % (BN / 4);
      constexpr uint stride_Bs = NUM_THREADS / (BN / 4);

      mywt::loadFromGmem<BM, BN, BK, stride_As, stride_Bs>(N, K, 
                      &A[global_block_row * K + k_base],
                      &B[k_base * N + global_block_col],
                      shmem_A, shmem_B,
                      as_row, as_col, bs_row, bs_col);
      __syncthreads();

      // compute using wmma
      for(uint k_inner = 0; k_inner < BK; k_inner += TK) {
          // load A and B matrix fragments
          wmma::fragment<wmma::matrix_a, TM, TN, TK, bf16, wmma::row_major> a_frag[TILES_PER_WARP_COL];
          wmma::fragment<wmma::matrix_b, TM, TN, TK, bf16, wmma::col_major> b_frag[TILES_PER_WARP_ROW];

          // matrix multiply-accumulate
          for(uint i = 0; i < TILES_PER_WARP_COL; ++i) {
              wmma::load_matrix_sync(a_frag[i],
                &shmem_A[(warpRow * WM + i * TM) * BK + k_inner], BK);
              for(uint j = 0; j < TILES_PER_WARP_ROW; ++j) {
                  if(i==0){
                    wmma::load_matrix_sync(b_frag[j],
                            &shmem_B[k_inner + (warpCol * WN + j * TN) * BK],
                            BK);
                  }
                  wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
              }
          }
      }
      __syncthreads();
    }

  // store fragments back to shared memory
  for(uint i = 0; i < TILES_PER_WARP_COL; ++i) {
    for(uint j = 0; j < TILES_PER_WARP_ROW; ++j) {
        for(uint t = 0; t < c_frag[i][j].num_elements; ++t) {
            c_frag[i][j].x[t] *= alpha;
        }
        wmma::store_matrix_sync(
            &shmem_C[(warpRow * WM + i * TM) * BN + warpCol * WN + j * TN],
                c_frag[i][j], BN, wmma::mem_row_major);
        }
 }
  __syncthreads();
    
  for(uint i = 0; i < BM / WARPS_PER_BLOCK; ++i) {
        float4 tmp = reinterpret_cast<const float4*>(
            &shmem_C[(shmem_C_row + i * WARPS_PER_BLOCK) * BN + shmem_C_col])[0];
        reinterpret_cast<float4*>(
            &C[(global_block_row + shmem_C_row + i * WARPS_PER_BLOCK) * N + global_block_col + shmem_C_col])[0] = tmp;
  }
  __syncthreads();

}


