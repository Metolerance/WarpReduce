#ifndef WARP_REDUCE_WITH_PADDING
#define WARP_REDUCE_WITH_PADDING

#include "common.cuh"

template<typename T, typename Op>
__device__ void warpLevelReduceWithPadding(int lane_id, int warp_id, int range, T value, T* warp_result, Op op);

template<typename T, typename Op>
__device__ T blockLevelReduceWithPadding(int idx, int n, T* warp_result, Op op);

template<typename T, typename Op>
__global__ void warpReduceWithPaddingKernel(int n, T* result, T* input_data, Op op);

template<typename T, typename Op>
T warpReduceWithPadding(int n, T* h_input, Op op);

#endif WARP_REDUCE_WITH_PADDING