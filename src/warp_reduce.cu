#include "../include/warp_reduce.cuh"

/*
 * warp内规约方法
 */
template<typename T, typename Op>
__device__ void warpLevelReduce(int lane_id, int warp_id, int range, T value, T* warp_result, Op op){
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
 * 块内规约方法1, 已废弃
 */
template<typename T, typename Op>
__device__ T blockLevelReduce1(int idx, int n, T* warp_result, T* block_result, Op op){

    for(int step = (n+1)/2; step>0; step = (step+1)/2){
        if((idx < step) && (idx + step < n)) {
            warp_result[idx] = op(warp_result[idx], warp_result[idx + step]);
        }
        // 块内同步
        __syncthreads();

        if (step==1)
            break;
    }

    // 块内0号线程执行存储操作
    if(threadIdx.x == 0) {
        block_result[blockIdx.x] = warp_result[0];
    }
    return warp_result[0];

}

/*
 * 块内规约方法2
 */
template<typename T, typename Op>
__device__ T blockLevelReduce(int idx, int n, T* warp_result, Op op){

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
 * 规约核函数， 已废弃
 */
template<typename T, typename Op>
__global__ void warpReduceKernel_1(int n, T* result, T* input_data, T* block_data, Op op){
    /*
     * 规约算法1，warp->块的循环，每块的结果放进设备内存再取出循环，没有原子操作，但是块间没同步时可能出错
     */

    int idx = blockDim.x * blockIdx.x + threadIdx.x;// 当前线程标号
    int block_num, warp_num, thread_num;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    T value = op.template identity<T>();

    if(idx < n)
        value = input_data[idx]; // 当前线程处理数据

    __shared__ T warp_data[BLOCK_DIM/WARP_SIZE];

    while(n>1){

        block_num = (n+WARP_SIZE-1) / WARP_SIZE; // 需要使用的块数
        if(blockIdx.x == block_num-1){
            warp_num = ((n%blockDim.x)+WARP_SIZE-1) / WARP_SIZE; // 当前块需要使用的warp数
        }
        else{
            warp_num = BLOCK_DIM / WARP_SIZE;
        }
        thread_num = WARP_SIZE > n ? n : WARP_SIZE; // 当前Warp需要使用的线程数

        if (idx < (n/WARP_SIZE + 1)*WARP_SIZE) {
            warpLevelReduce(lane_id, warp_id, thread_num, value, warp_data, op);
            if (n > WARP_SIZE) {
                blockLevelReduce1(threadIdx.x, warp_num, warp_data, block_data, op);
            }
            else {
                value = warp_data[0];
                break;
            }

            n = (n + (int) blockDim.x - 1) / (int) blockDim.x;

            if (idx < n) {
                value = block_data[idx];
            }
        }

    }

    if (idx == 0)
        result[0] = value;

}

/*
 * 规约核函数
 */
template<typename T, typename Op>
__global__ void warpReduceKernel(int n, T* result, T* input_data, Op op){
    /*
     * 规约算法2，warp->块的循环，每块的结果用原子操作放进设备内存
     */

    int idx = blockDim.x * blockIdx.x + threadIdx.x;// 当前线程标号
    int block_num, warp_num, thread_num;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    T value = op.template identity<T>();

    if(idx < n)
        value = input_data[idx]; // 当前线程处理数据

    __shared__ T warp_data[BLOCK_DIM/WARP_SIZE];

    block_num = (n+WARP_SIZE-1) / WARP_SIZE; // 需要使用的块数
    if(blockIdx.x == block_num-1){
        warp_num = ((n%blockDim.x)+WARP_SIZE-1) / WARP_SIZE; // 当前块需要使用的warp数
    }
    else{
        warp_num = BLOCK_DIM / WARP_SIZE;
    }
    thread_num = WARP_SIZE > n ? n : WARP_SIZE; // 当前Warp需要使用的线程数

    if (idx < (n/WARP_SIZE + 1)*WARP_SIZE) {
        warpLevelReduce(lane_id, warp_id, thread_num, value, warp_data, op);
        __syncthreads();
        if (n > WARP_SIZE) {
            value = blockLevelReduce(threadIdx.x, warp_num, warp_data, op);
        }
        else {
            value = warp_data[0];
        }
    }

    if (threadIdx.x == 0){
        atomicAdd(result, value);
    }
}

/*
 * 规约方法
 */
template<typename T, typename Op>
T warpReduce(int n, T* h_input, Op op){

    // 声明变量
    T h_result = 0;
    T* d_intput, *d_result;

    // 分配设备内存
    CUDA_CHECK(cudaMalloc(&d_intput, (n)*sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(T)));

    // 复制主机内存到设备内存
    CUDA_CHECK(cudaMemcpy(d_intput, h_input, n*sizeof(T), cudaMemcpyHostToDevice));

    // 调用CUB核函数
//    void *d_temp_storage = nullptr;
//    size_t temp_storage_bytes = 0;
//    cub::DeviceReduce::Sum( d_temp_storage, temp_storage_bytes, d_intput, d_result, n);
//    cudaMalloc(&d_temp_storage, temp_storage_bytes);
//    cub::DeviceReduce::Sum( d_temp_storage, temp_storage_bytes, d_intput, d_result, n);
    // 调用核函数
    warpReduceKernel<<<ceil((float) n / BLOCK_DIM), BLOCK_DIM>>>(n, d_result, d_intput, op);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 复制结果到主机内存
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost));

    // 释放内存
    CUDA_CHECK(cudaFree(d_intput));
    CUDA_CHECK(cudaFree(d_result));

    return h_result;
}

// 模板显式实例化
template float warpReduce<float, SumOp>(int n, float *h_input, SumOp op);
