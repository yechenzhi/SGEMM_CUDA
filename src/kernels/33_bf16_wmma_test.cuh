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

namespace {
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
__global__ void bf16_wmma_test(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
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
  
  //used for computing wmma within the warptile
  //here we simply use 4 * 4 warptiles with a block
  const uint warp_row = threadIdx.x / WARPSIZE; // [0,1,2,3]
  const uint warp_col = threadIdx.y; //[0,1,2,3]
  
  wmma::fragment<wmma::matrix_a, TM, TN, TK, __nv_bfloat16, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, TM, TN, TK, __nv_bfloat16, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, TM, TN, TK, float> acc_frag;
  wmma::fragment<wmma::accumulator, TM, TN, TK, float> c_frag;

  wmma::fill_fragment(acc_frag, 0.0f);

  __shared__ bf16 As[BM * BK];
  __shared__ bf16 Bs[BN * BK]; // BN x BK row major can be treated as BK x BN col major

  A += row_start_block * K;
  B += col_start_block;
  C += row_start_block * N + col_start_block;

  for (uint k_base = 0; k_base < K; k_base += BK) {
    // load A and B to shared memory
    loadFromGmem<BM, BN, BK, stride_As, stride_Bs>(N, K, A, B, 
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

        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down
    __syncthreads();
  }

  const uint ldc = N;
  const uint c_row = warp_row * TM;
  const uint c_col = warp_col * TN;
  if ( c_row < M && c_col < N ) {
    wmma::load_matrix_sync(c_frag, &C[c_row * ldc + c_col], ldc, wmma::mem_row_major);

    for (int i = 0; i < c_frag.num_elements; i++) {
      c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
    }

    wmma::store_matrix_sync(&C[c_row * ldc + c_col], c_frag, ldc, wmma::mem_row_major);
  }

}

