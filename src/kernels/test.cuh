const int row_start_block = blockIdx.y * BM;
  const int col_start_block = blockIdx.x * BN;
  
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];
  float threadResults[WMITER*TM*WNITER*TN] = {0.0};
  float regM_inner[WMITER * TM];
  float regN_inner[WNITER * TN];

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