#include <helper_cuda.h>

template <typename T>
__global__ void sum_kernel (T* input, int N, T initial) {
    auto sum = initial;
    for (int i = 0; i < N; ++i)
        sum += input[i];


    input[0] = sum;
}

template <typename T>
void sum(T* input, int N, T initial) {
    int blockSize = 1024;
    int gridSize = (N + blockSize - 1) / blockSize;
    dim3 blockDim(blockSize);
    dim3 gridDim(gridSize);


    sum_kernel<<<gridDim, blockDim>>>(input, N, initial);
}

template 
void sum<float> (float*, int, float);

template 
__global__ void sum_kernel<float> (float*, int, float);
//template <typename T>
//void sum_void sum_atomic_kernel (T* input, int N, T initial) {
//    auto sum = initial;
//    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
//    if (i < N) {
//        atomicAdd(sum, input[i]);
//    }
//
//
//    input[0] = sum;
//}
//
//template <typename T>
//void sum_atomic(T* input, int N, T initial) {
//    int blockSize = 1024;
//    int gridSize = (N + blockSize - 1) / blockSize;
//    dim3 blockDim(blockSize);
//    dim3 gridDim(gridSize);
//
//
//    sum_kernel<<<gridDim, blockDim>>>(input, N, initial);
//}
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
