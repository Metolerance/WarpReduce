#include "../include/warp_reduce.cuh"

/*
 * warp�ڹ�Լ����
 */
template<typename T, typename Op>
__device__ void warpLevelReduce(int lane_id, int warp_id, int range, T value, T* warp_result, Op op){
    // ѭ����Լ
    #pragma unroll
    for(int step=(range+1)/2; step>0; range=step, step = (step+1)/2){
        // �߳�ͨ��
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
 * ���ڹ�Լ����1
 */
template<typename T, typename Op>
__device__ T blockLevelReduce1(int idx, int n, T* warp_result, T* block_result, Op op){

    for(int step = (n+1)/2; step>0; step = (step+1)/2){
        if((idx < step) && (idx + step < n)) {
            warp_result[idx] = op(warp_result[idx], warp_result[idx + step]);
        }
        // ����ͬ��
        __syncthreads();

        if (step==1)
            break;
    }

    // ����0���߳�ִ�д洢����
    if(threadIdx.x == 0) {
        block_result[blockIdx.x] = warp_result[0];
    }
    return warp_result[0];

}

/*
 * ���ڹ�Լ����2
 */
template<typename T, typename Op>
__device__ T blockLevelReduce2(int idx, int n, T* warp_result, Op op){

    for(int step = (n+1)/2; step>0; step = (step+1)/2){
        if((idx < step) && (idx + step < n)) {
            warp_result[idx] = op(warp_result[idx], warp_result[idx + step]);
        }
        // ����ͬ��
        __syncthreads();

        if (step==1)
            break;
    }

    return warp_result[0];
}


/*
 * ��Լ�˺���
 */
template<typename T, typename Op>
__global__ void warpReduceKernel_1(int n, T* result, T* input_data, T* block_data, Op op){
    /*
     * ��Լ�㷨1��warp->���ѭ����ÿ��Ľ���Ž��豸�ڴ���ȡ��ѭ����û��ԭ�Ӳ��������ǿ��ûͬ��ʱ���ܳ���
     */

    int idx = blockDim.x * blockIdx.x + threadIdx.x;// ��ǰ�̱߳��
    int block_num, warp_num, thread_num;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    T value = op.template identity<T>();

    if(idx < n)
        value = input_data[idx]; // ��ǰ�̴߳�������

    __shared__ T warp_data[BLOCK_DIM/WARP_SIZE];

    while(n>1){

        block_num = (n+WARP_SIZE-1) / WARP_SIZE; // ��Ҫʹ�õĿ���
        if(blockIdx.x == block_num-1){
            warp_num = ((n%blockDim.x)+WARP_SIZE-1) / WARP_SIZE; // ��ǰ����Ҫʹ�õ�warp��
        }
        else{
            warp_num = BLOCK_DIM / WARP_SIZE;
        }
        thread_num = WARP_SIZE > n ? n : WARP_SIZE; // ��ǰWarp��Ҫʹ�õ��߳���

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
 * ��Լ�˺���
 */
template<typename T, typename Op>
__global__ void warpReduceKernel_2(int n, T* result, T* input_data, Op op){
    /*
     * ��Լ�㷨2��warp->���ѭ����ÿ��Ľ����ԭ�Ӳ����Ž��豸�ڴ�
     */

    int idx = blockDim.x * blockIdx.x + threadIdx.x;// ��ǰ�̱߳��
    int block_num, warp_num, thread_num;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    T value = op.template identity<T>();

    if(idx < n)
        value = input_data[idx]; // ��ǰ�̴߳�������

    __shared__ T warp_data[BLOCK_DIM/WARP_SIZE];

    block_num = (n+WARP_SIZE-1) / WARP_SIZE; // ��Ҫʹ�õĿ���
    if(blockIdx.x == block_num-1){
        warp_num = ((n%blockDim.x)+WARP_SIZE-1) / WARP_SIZE; // ��ǰ����Ҫʹ�õ�warp��
    }
    else{
        warp_num = BLOCK_DIM / WARP_SIZE;
    }
    thread_num = WARP_SIZE > n ? n : WARP_SIZE; // ��ǰWarp��Ҫʹ�õ��߳���

    if (idx < (n/WARP_SIZE + 1)*WARP_SIZE) {
        warpLevelReduce(lane_id, warp_id, thread_num, value, warp_data, op);
        if (n > WARP_SIZE) {
            value = blockLevelReduce2(threadIdx.x, warp_num, warp_data, op);
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
 * ��Լ����
 */
template<typename T, typename Op>
T warpReduce(int n, T* h_input, Op op){

    // ��������
    T h_result = 0;
    T* d_intput, *d_result, *d_block_data;

    // �����豸�ڴ�
    CUDA_CHECK(cudaMalloc(&d_intput, n*sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(T)));
//    CUDA_CHECK(cudaMalloc(&d_block_data, GRID_DIM*sizeof(T)));

    // ���������ڴ浽�豸�ڴ�
    CUDA_CHECK(cudaMemcpy(d_intput, h_input, n*sizeof(T), cudaMemcpyHostToDevice));

    // ����CUB�˺���
//    void *d_temp_storage = nullptr;
//    size_t temp_storage_bytes = 0;
//    cub::DeviceReduce::Sum( d_temp_storage, temp_storage_bytes, d_intput, d_result, n);
//    cudaMalloc(&d_temp_storage, temp_storage_bytes);
//    cub::DeviceReduce::Sum( d_temp_storage, temp_storage_bytes, d_intput, d_result, n);
    // ���ú˺���
    warpReduceKernel_2<<<ceil((float)n/BLOCK_DIM), BLOCK_DIM>>>(n, d_result, d_intput, op);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ���ƽ���������ڴ�
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost));

    // �ͷ��ڴ�
    CUDA_CHECK(cudaFree(d_intput));
    CUDA_CHECK(cudaFree(d_result));
//    CUDA_CHECK(cudaFree(d_block_data));

    return h_result;
}

// ģ����ʽʵ����
template float warpReduce<float, SumOp>(int n, float *h_input, SumOp op);
template float warpReduce<float, MaxOp>(int n, float *h_input, MaxOp op);
template float warpReduce<float, MinOp>(int n, float *h_input, MinOp op);