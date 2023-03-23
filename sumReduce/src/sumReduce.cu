#include <helper_cuda.h>
#include <cstdio>

template <typename T>
__global__ void sum_kernel (T* input, int N, T initial) {
    auto sum = initial;
    for (int i = 0; i < N; ++i)
        sum += input[i];

    input[0] = sum;
}

template <typename T>
__global__ void sum_atomic_kernel (T* input, int N, T* initial) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) {
        atomicAdd(initial, input[i]);
    }
}

template <typename T>
__global__ void sum_seg_reduction_kernel(T* input) {
    unsigned int segment = 2*blockDim.x*blockIdx.x;
    unsigned int i = segment + 2*threadIdx.x;
    for(unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) {
            input[i] += input [i + stride];
        }
        __syncthreads();
    }
}

template <typename T>
void sum_seg_reduction(T* input, int N){
    int blockSize = 2;
    int gridSize = (N/2)+ (blockSize - 1) / blockSize;
    dim3 blockDim(blockSize);
    dim3 gridDim(gridSize);

    sum_seg_reduction_kernel<<<gridDim, blockDim>>>(input);
    checkCudaErrors(cudaGetLastError());
}

template <typename T>
void sum(T* input, int N, T initial) {
    sum_kernel<<<1,1>>>(input, N, initial);
    checkCudaErrors(cudaGetLastError());
}

template <typename T>
void sum_atomic(T* input, int N, T* initial) {
    int blockSize = 32;
    int gridSize = (N + blockSize - 1) / blockSize;
    dim3 blockDim(blockSize);
    dim3 gridDim(gridSize);

    sum_atomic_kernel<<<gridDim, blockDim>>>(input, N, initial);
    checkCudaErrors(cudaGetLastError());
}

template void sum<float> (float*, int, float);
template __global__ void sum_kernel<float> (float*, int, float);

template void sum_atomic<float> (float*, int, float*);
template __global__ void sum_atomic_kernel<float> (float*, int, float*);

template __global__ void sum_seg_reduction_kernel<float> (float*);
template void sum_seg_reduction<float> (float*, int);
//
//template <typename T>
//void sum_atomic_kernel(T* input, int N) {
//    unsigned int segment = 2*blockDim.x*blockIdx.x;
//    unsigned int i = segment + 2*threadIdx.x;
//    for (threadIdx.x%stride == 0) {
//        input[i] += input[i + stride];
//    }
//    __syncthreads();
//}
//
