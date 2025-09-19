#include "../include/warp_reduce_with_padding.cuh"

/*
 * warp�ڹ�Լ����
 */
template<typename T, typename Op>
__device__ void warpLevelReduceWithPadding(int lane_id, int warp_id, int range, T value, T* warp_result, Op op){
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
 * ���ڹ�Լ����
 */
template<typename T, typename Op>
__device__ T blockLevelReduceWithPadding(int idx, int n, T* warp_result, Op op) {

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
 * ģ��ԭ�Ӳ���, TODO
 */
template<typename T, typename Op>
__device__ void atomicOperation(T* result, T value, Op op){
    // �ж�T��Op����ʹ��ʲôԭ�Ӳ������й�Լ

    return;
}

/*
 * ��Լ�˺���
 */
template<typename T, typename Op>
__global__ void warpReduceWithPaddingKernel(int n, T* result, T* input_data, Op op){
    /*
     * ��Լ�㷨��warp->���ѭ����ÿ��Ľ����ԭ�Ӳ����Ž��豸�ڴ�
     */

    int idx = blockDim.x * blockIdx.x + threadIdx.x;// ��ǰ�̱߳��
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
 * ��Լ������Ŀǰ��֧�ּӷ�
 */
template<typename T, typename Op>
T warpReduceWithPadding(int n, T* h_input, Op op){

    // ��������
    T h_result = 0;
    T* d_intput, *d_result, *padding_data;

    // �������Ԫ��
    unsigned int padding_num = (BLOCK_DIM - (n%BLOCK_DIM)) % BLOCK_DIM;
    padding_data = new T[padding_num];
    std::fill(padding_data, padding_data+padding_num, op.template identity<T>());

    // �����豸�ڴ�
    CUDA_CHECK(cudaMalloc(&d_intput, (n+padding_num)*sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(T)));

    // ���������ڴ浽�豸�ڴ�
    CUDA_CHECK(cudaMemcpy(d_intput, h_input, n*sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_intput+n, padding_data, padding_num*sizeof(T), cudaMemcpyHostToDevice));

    // ���ú˺���
    warpReduceWithPaddingKernel<<<ceil((float) n / BLOCK_DIM), BLOCK_DIM>>>(n, d_result, d_intput, op);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ���ƽ���������ڴ�
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost));

    // �ͷ��ڴ�
    CUDA_CHECK(cudaFree(d_intput));
    CUDA_CHECK(cudaFree(d_result));
    delete[] padding_data;

    return h_result;
}

// ģ����ʽʵ����
template float warpReduceWithPadding<float, SumOp>(int n, float *h_input, SumOp op);
