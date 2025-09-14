#ifndef WARP_REDUCE
#define WARP_REDUCE

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <climits>
#include <cub/cub.cuh>

#define WARP_SIZE 32
#define BLOCK_DIM 128
#define GRID_DIM 8

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

struct MaxOp {
    template <typename T>
    __host__ __device__ T operator()(T a, T b) const {
        return a > b ? a : b;
    }
    template <typename T>
    __host__ __device__ T identity() const{
        return -FLT_MAX;
    }
};

struct MinOp {
    template <typename T>
    __host__ __device__ T operator()(T a, T b) const {
        return a < b ? a : b;
    }
    template <typename T>
    __host__ __device__ T identity() const{
        return FLT_MAX;
    }
};

struct SumOp {
    template <typename T>
    __host__ __device__ T operator()(T a, T b) const {
        return a + b;
    }
    template <typename T>
    __host__ __device__ T identity() const{
        return 0;
    }
};

template<typename T, typename Op>
__device__ void warpLevelReduce(int lane_id, int warp_id, int range, T value, T* warp_result, Op op);

template<typename T, typename Op>
__device__ T blockLevelReduce1(int idx, int n, T* warp_result, T* block_result, Op op);
template<typename T, typename Op>
__device__ T blockLevelReduce2(int idx, int n, T* warp_result, Op op);

template<typename T, typename Op>
__device__ void warpLevelReduce(int lane_id, int warp_id, int range, T value, T* warp_result, Op op);

template<typename T, typename Op>
__global__ void warpReduceKernel_2(int n, T* result, T* input_data, Op op);

template<typename T, typename Op>
T warpReduce(int n, T* h_input, Op op);

#endif WARP_REDUCE