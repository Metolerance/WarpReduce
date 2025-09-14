#include "warp_reduce.cuh"

#define N 10000

int main(){

    float h_input[N], result;
    for(int i=0; i<N; i++){
        h_input[i] = 1;
    }
    result = warpReduce(N, h_input, SumOp());

    printf("result: %.0f", result);
    return 0;
}

