#ifndef TEST_COMMON_CUH
#define TEST_COMMON_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <climits>
#include <cub/cub.cuh>

#define WARP_SIZE 32
#define BLOCK_DIM 128

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)


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

#endif //TEST_COMMON_CUH
