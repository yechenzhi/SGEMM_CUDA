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
using bf162 = __nv_bfloat162;
using namespace nvcuda;
// const int WARPSIZE = 32; // warpSize is not constexpr

/*

Matrix sizes:
MxK(row major) * KxN(row major) = MxN

*/

namespace bf16_warptiling_float2{
    template <const int BM, const int BN, const int BK, const int stride_As, const int stride_Bs>
    __device__ void loadFromGmem(int N, int K, const float *A, const float *B,
                                 bf16 *As, bf16 *Bs,
                                 int as_row, int as_col, int bs_row, int bs_col) {

        for(uint row_offset = 0; row_offset < BM; row_offset += stride_As) {
            float4 tmp_fp32 =
                reinterpret_cast<const float4 *>(&A[(row_offset + as_row) * K + as_col * 4])[0];
            bf16* target = &As[(row_offset + as_row) * BK + as_col * 4];

            float2 low = make_float2(tmp_fp32.x, tmp_fp32.y);
            float2 high = make_float2(tmp_fp32.z, tmp_fp32.w);
            bf162 res_low = __float22bfloat162_rn(low);
            bf162 res_high = __float22bfloat162_rn(high);
            reinterpret_cast<bf162*>(target)[0] = res_low;
            reinterpret_cast<bf162*>(target)[1] = res_high;
        }
        for(uint row_offset = 0; row_offset < BK; row_offset += stride_Bs) {
            float4 tmp_fp32 = 
                reinterpret_cast<const float4 *>(&B[(row_offset + bs_row) * N + bs_col * 4])[0];
            bf16* target = &Bs[(row_offset + bs_row) * BN + bs_col * 4];
            
            float2 low = make_float2(tmp_fp32.x, tmp_fp32.y);
            float2 high = make_float2(tmp_fp32.z, tmp_fp32.w);
            bf162 res_low = __float22bfloat162_rn(low);
            bf162 res_high = __float22bfloat162_rn(high);
            reinterpret_cast<bf162*>(target)[0] = res_low;
            reinterpret_cast<bf162*>(target)[1] = res_high;
        }
    }
} 

template <const int BM, const int BN, const int BK,const int WM, const int WN, const int TM, const int TN, const int TK,
          const int NUM_THREADS>
__global__ void bf16_wmma_warptiling_float2(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  beta /= alpha;
  const uint tid = threadIdx.y * blockDim.x + threadIdx.x;
  const uint row_start_block = blockIdx.x * BM;
  const uint col_start_block = blockIdx.y * BN;

  // for loading A and B to shared memory
  const uint as_row = tid / (BK / 4);
  const uint as_col = tid % (BK / 4);
  const uint stride_As = NUM_THREADS / (BK / 4);
  const uint bs_row = tid / (BN / 4);
  const uint bs_col = tid % (BN / 4);
  const uint stride_Bs = NUM_THREADS / (BN / 4);
  const uint cs_row = tid / (BN / 4);
  const uint cs_col = tid % (BN / 4);
  const uint stride_Cs = NUM_THREADS / (BN / 4);
  
  //used for warp index with a block
  //here we simply use 4 * 2 warps with a block
//   const uint warp_row = threadIdx.x / WARPSIZE; // [0,1]
//   const uint warp_col = threadIdx.y; //[0,1,2,3]

  // suggest by gpt
  const int WARPS_PER_BLOCK = NUM_THREADS / WARPSIZE;
  const int WARPS_PER_BLOCK_M = BM / WM;               //128 / 32 = 4
  const int WARPS_PER_BLOCK_N = BN / WN;               // 2
  int warp_id = tid / WARPSIZE;
  int warp_row = warp_id / WARPS_PER_BLOCK_N;
  int warp_col = warp_id % WARPS_PER_BLOCK_N;  

  //used for tile index with a warp
  //here we simple use 2 * 4 tiles with a warp
  const uint TILES_PER_WARP_ROW = WN / TN; // 64 / 16 = 4
  const uint TILES_PER_WARP_COL = WM / TM; // 32 / 16 = 2

  
//   wmma::fragment<wmma::accumulator, TM, TN, TK, float> acc_frag;
  wmma::fragment<wmma::matrix_a, TM, TN, TK, __nv_bfloat16, wmma::row_major> a_frag[TILES_PER_WARP_COL];
  wmma::fragment<wmma::matrix_b, TM, TN, TK, __nv_bfloat16, wmma::row_major> b_frag[TILES_PER_WARP_ROW];
  wmma::fragment<wmma::accumulator, TM, TN, TK, float> c_frag[TILES_PER_WARP_COL][TILES_PER_WARP_ROW];

  extern __shared__ char shem[];
  bf16* As = (bf16*)shem;
  bf16* Bs = As + BM * BK;
  float* Cs = (float*)shem;

  A += row_start_block * K;
  B += col_start_block;
  C += row_start_block * N + col_start_block;

  const uint ldc = BN;
  const uint cRow_start = warp_row * WM;
  const uint cCol_start = warp_col * WN;

  for(uint row_offset = 0; row_offset < BM; row_offset += stride_Cs) {
        float4 tmp_fp32 =
            reinterpret_cast<const float4 *>(&C[(row_offset + cs_row) * N + cs_col * 4])[0];
        reinterpret_cast<float4*>(&Cs[(row_offset + cs_row) * BN + cs_col * 4])[0] = tmp_fp32;
  }
   __syncthreads();

  for(uint i = 0; i < TILES_PER_WARP_COL; i++){
    for(uint j = 0; j < TILES_PER_WARP_ROW; j++){
        const uint tile_pos = (cRow_start + i * TM) * ldc + cCol_start + j * TN;
        wmma::load_matrix_sync(c_frag[i][j], &Cs[tile_pos], ldc, wmma::mem_row_major);
    }
  }
  __syncthreads();

  for(uint i = 0; i < TILES_PER_WARP_COL; i++){
    for(uint j = 0; j < TILES_PER_WARP_ROW; j++){
        for (int t = 0; t < c_frag[i][j].num_elements; t++) {
            c_frag[i][j].x[t] *= beta;
        }
    }
  }

  for (uint k_base = 0; k_base < K; k_base += BK) {
    // load A and B to shared memory
    bf16_warptiling_float2::loadFromGmem<BM, BN, BK, stride_As, stride_Bs>(N, K, A, B, 
                    As, Bs, as_row, as_col, bs_row, bs_col);
    __syncthreads();

    for(uint k_inner = 0; k_inner < BK; k_inner += TK) {
        const uint lda = BK;
        const uint ldb = BN;

        const uint aRow_start = warp_row * WM;
        const uint aCol_start = k_inner;
        const uint bRow_start = k_inner;
        const uint bCol_start = warp_col * WN;
    
        for(uint i = 0; i < TILES_PER_WARP_COL; i++){
            wmma::load_matrix_sync(a_frag[i], &As[(aRow_start + i * TM) * lda + aCol_start], lda);
            for(uint j = 0; j < TILES_PER_WARP_ROW; j++){
                if(i==0){
                    wmma::load_matrix_sync(b_frag[j], &Bs[bRow_start * ldb + bCol_start + j * TN], ldb);
                }
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
            }
        }
    }

    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down
    __syncthreads();
  }

  // store the D fragements to shared memory
  for(uint i = 0; i < TILES_PER_WARP_COL; i++){
    for(uint j = 0; j < TILES_PER_WARP_ROW; j++){
        for (int t = 0; t < c_frag[i][j].num_elements; t++)
            c_frag[i][j].x[t] *= alpha; 
        const uint tile_pos = (cRow_start + i * TM) * ldc + cCol_start + j * TN;
        wmma::store_matrix_sync(&Cs[tile_pos], c_frag[i][j], ldc, wmma::mem_row_major);
    }
  }
  __syncthreads();
  
  for(uint row_offset = 0; row_offset < BM; row_offset += stride_Cs) {
        float4 tmp =
            reinterpret_cast<const float4 *>(&Cs[(row_offset + cs_row) * BN + cs_col * 4])[0];
        reinterpret_cast<float4*>(&C[(row_offset + cs_row) * N + cs_col * 4])[0] = tmp;
    }
}