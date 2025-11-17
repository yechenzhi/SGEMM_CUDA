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

/*

Matrix sizes:
MxK(row major) * KxN(row major) = MxN

*/

namespace bf16_reuse{
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
            target_addr[0] = __float2bfloat16(tmp_fp32.x);
            target_addr[1] = __float2bfloat16(tmp_fp32.y);
            target_addr[2] = __float2bfloat16(tmp_fp32.z);
            target_addr[3] = __float2bfloat16(tmp_fp32.w);
        }
    }
} 

template <const int BM, const int BN, const int BK, const int TM, const int TN, const int TK,
          const int NUM_THREADS>
__global__ void bf16_wmma_reuse(int M, int N, int K, float alpha, const float *A,
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
  
  //used for computing wmma within the warptile
  //here we simply use 4 * 4 warptiles with a block
  const uint warp_row = threadIdx.x / WARPSIZE; // [0,1,2,3]
  const uint warp_col = threadIdx.y; //[0,1,2,3]
  
  wmma::fragment<wmma::matrix_a, TM, TN, TK, __nv_bfloat16, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, TM, TN, TK, __nv_bfloat16, wmma::row_major> b_frag;
//   wmma::fragment<wmma::accumulator, TM, TN, TK, float> acc_frag;
  wmma::fragment<wmma::accumulator, TM, TN, TK, float> c_frag;

  extern __shared__ char shem[];
  bf16* As = (bf16*)shem;
  bf16* Bs = As + BM * BK;
  float* Cs = (float*)shem;

  A += row_start_block * K;
  B += col_start_block;
  C += row_start_block * N + col_start_block;

  const uint ldc = BN;
  const uint cRow = warp_row * TM;
  const uint cCol = warp_col * TN;

  for(uint row_offset = 0; row_offset < BM; row_offset += stride_Cs) {
        float4 tmp_fp32 =
            reinterpret_cast<const float4 *>(&C[(row_offset + cs_row) * N + cs_col * 4])[0];
        reinterpret_cast<float4*>(&Cs[(row_offset + cs_row) * BN + cs_col * 4])[0] = tmp_fp32;
  }
   __syncthreads();

  wmma::load_matrix_sync(c_frag, &Cs[cRow * ldc + cCol], ldc, wmma::mem_row_major);
  __syncthreads();

  for (int t = 0; t < c_frag.num_elements; t++) {
    c_frag.x[t] *= beta;
  }

  for (uint k_base = 0; k_base < K; k_base += BK) {
    // load A and B to shared memory
    bf16_reuse::loadFromGmem<BM, BN, BK, stride_As, stride_Bs>(N, K, A, B, 
                    As, Bs, as_row, as_col, bs_row, bs_col);
    __syncthreads();

    for(uint k_inner = 0; k_inner < BK; k_inner += TK) {
        const uint lda = BK;
        const uint ldb = BN;

        const uint aRow = warp_row * TM;
        const uint aCol = k_inner;
        const uint bRow = k_inner;
        const uint bCol = warp_col * TN;

        wmma::load_matrix_sync(a_frag, &As[aRow * lda + aCol], lda);
        wmma::load_matrix_sync(b_frag, &Bs[bRow * ldb + bCol], ldb);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down
    __syncthreads();
  }

  // store the D fragements to shared memory
  for (int t = 0; t < c_frag.num_elements; t++)
    c_frag.x[t] *= alpha;  
  wmma::store_matrix_sync(&Cs[cRow * ldc + cCol], c_frag, ldc, wmma::mem_row_major);
  __syncthreads();
  
  for(uint row_offset = 0; row_offset < BM; row_offset += stride_Cs) {
        float4 tmp =
            reinterpret_cast<const float4 *>(&Cs[(row_offset + cs_row) * BN + cs_col * 4])[0];
        reinterpret_cast<float4*>(&C[(row_offset + cs_row) * N + cs_col * 4])[0] = tmp;
    }
}

