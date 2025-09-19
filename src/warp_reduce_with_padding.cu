#include "../include/warp_reduce_with_padding.cuh"

/*
 * warp内规约方法
 */
template<typename T, typename Op>
__device__ void warpLevelReduceWithPadding(int lane_id, int warp_id, int range, T value, T* warp_result, Op op){
    // 循环规约
    #pragma unroll
    for(int step=(range+1)/2; step>0; range=step, step = (step+1)/2){
        // 线程通信
        if(lane_id < range)
            value = op(value, __shfl_down_sync(__activemask(), value, step));

        if (step==1)
            break;
    }

    if(lane_id == 0) {
        warp_result[warp_id] = value;
    }
}

/*
 * 块内规约方法
 */
template<typename T, typename Op>
__device__ T blockLevelReduceWithPadding(int idx, int n, T* warp_result, Op op) {

    for(int step = (n+1)/2; step>0; step = (step+1)/2){
        if((idx < step) && (idx + step < n)) {
            warp_result[idx] = op(warp_result[idx], warp_result[idx + step]);
        }
        // 块内同步
        __syncthreads();

        if (step==1)
            break;
    }

    return warp_result[0];
}

/*
 * 模板原子操作, TODO
 */
template<typename T, typename Op>
__device__ void atomicOperation(T* result, T value, Op op){
    // 判定T和Op决定使用什么原子操作进行规约

    return;
}

/*
 * 规约核函数
 */
template<typename T, typename Op>
__global__ void warpReduceWithPaddingKernel(int n, T* result, T* input_data, Op op){
    /*
     * 规约算法，warp->块的循环，每块的结果用原子操作放进设备内存
     */

    int idx = blockDim.x * blockIdx.x + threadIdx.x;// 当前线程标号
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    T value = input_data[idx];

    __shared__ T warp_data[BLOCK_DIM/WARP_SIZE];

    warpLevelReduceWithPadding(lane_id, warp_id, WARP_SIZE, value, warp_data, op);

    __syncthreads();

    value = blockLevelReduceWithPadding(threadIdx.x, blockDim.x, warp_data, op);

    if (threadIdx.x == 0){
        atomicAdd(result, value);
    }
}

/*
 * 规约方法，目前仅支持加法
 */
template<typename T, typename Op>
T warpReduceWithPadding(int n, T* h_input, Op op){

    // 声明变量
    T h_result = 0;
    T* d_intput, *d_result, *padding_data;

    // 计算填充元素
    unsigned int padding_num = (BLOCK_DIM - (n%BLOCK_DIM)) % BLOCK_DIM;
    padding_data = new T[padding_num];
    std::fill(padding_data, padding_data+padding_num, op.template identity<T>());

    // 分配设备内存
    CUDA_CHECK(cudaMalloc(&d_intput, (n+padding_num)*sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(T)));

    // 复制主机内存到设备内存
    CUDA_CHECK(cudaMemcpy(d_intput, h_input, n*sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_intput+n, padding_data, padding_num*sizeof(T), cudaMemcpyHostToDevice));

    // 调用核函数
    warpReduceWithPaddingKernel<<<ceil((float) n / BLOCK_DIM), BLOCK_DIM>>>(n, d_result, d_intput, op);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 复制结果到主机内存
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost));

    // 释放内存
    CUDA_CHECK(cudaFree(d_intput));
    CUDA_CHECK(cudaFree(d_result));
    delete[] padding_data;

    return h_result;
}

// 模板显式实例化
template float warpReduceWithPadding<float, SumOp>(int n, float *h_input, SumOp op);
