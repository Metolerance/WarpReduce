#ifndef WARP_REDUCE
#define WARP_REDUCE

#include "common.cuh"

template<typename T, typename Op>
__device__ void warpLevelReduce(int lane_id, int warp_id, int range, T value, T* warp_result, Op op);

template<typename T, typename Op>
__device__ T blockLevelReduce1(int idx, int n, T* warp_result, T* block_result, Op op);
template<typename T, typename Op>
__device__ T blockLevelReduce(int idx, int n, T* warp_result, Op op);

template<typename T, typename Op>
__device__ void warpLevelReduce(int lane_id, int warp_id, int range, T value, T* warp_result, Op op);

template<typename T, typename Op>
__global__ void warpReduceKernel(int n, T* result, T* input_data, Op op);

template<typename T, typename Op>
T warpReduce(int n, T* h_input, Op op);

#endif WARP_REDUCE