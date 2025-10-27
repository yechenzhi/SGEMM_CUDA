#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BLOCKSIZE>
__global__ void my_sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  
  __shared__ float As[BLOCKSIZE][BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

  // the inner row & col that we're accessing in this thread
  const uint threadCol = threadIdx.x;
  const uint threadRow = threadIdx.y;


  float tmp = 0.0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // Have each thread load one of the elements in A & B
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing

    // for A: row = row, col = bkIdx + threadCol
    // for B: row = bkIdx + threadRow, col = col
    if (row < M && (bkIdx + threadCol) < K) {
        As[threadRow][threadCol] = A[row * K + (bkIdx + threadCol)];
    } else {
        As[threadRow][threadCol] = 0.0;
    }

    if ((bkIdx + threadRow) < K && col < N) {
        Bs[threadRow][threadCol] = B[(bkIdx + threadRow) * N + col];
    } else {
        Bs[threadRow][threadCol] = 0.0;
    }

    // block threads in this block until cache is fully populated
    __syncthreads();

    // execute the dotproduct on the currently cached block
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += As[threadRow][dotIdx] *
             Bs[dotIdx][threadCol];
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }
  C[row * N + col] =
      alpha * tmp + beta * C[row * N + col];
}