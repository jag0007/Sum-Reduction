#include <helper_cuda.h>
#include <cstdio>

template <typename T>
__global__ void sum_kernel (T* input, int N, T* initial) {
    auto sum = *initial;
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
__global__ void sum_seg_reduction_kernel(T* input, int N, T* output) {
    unsigned int segment = 2*blockDim.x*blockIdx.x;
    unsigned int i = segment + 2*threadIdx.x;
    for(unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) {
            input[i] += input [i + stride];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        atomicAdd(output, input[i]);
    }
}

template <typename T>
__global__ void sum_coalecsing_kernel(T* input, int N, T* output) {
    unsigned int segment = 2*blockDim.x*blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    for(unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
        if(threadIdx.x <  stride) {
            input[i] += input[i + stride];
        }
    __syncthreads();
    }
    if(threadIdx.x ==0){
        atomicAdd(output, input[i]);
    }
}

template <typename T>
__global__ void sum_shared_mem_kernel(T* input, int N, T* output) {
    unsigned int segment = 2*blockDim.x*blockIdx.x;
    unsigned int i = segment + threadIdx.x;

    // Load data to shared memory
    extern __shared__ float input_s[];
    input_s[threadIdx.x] = input[i] + input[i + blockDim.x];
    __syncthreads();

    // Reduction tree in shared memory
    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
        if(threadIdx.x < stride) {
            input_s[threadIdx.x] += input_s[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        atomicAdd(output, input_s[0]);
    }
}

template <typename T>
__global__ void sum_coarsened_kernel(T* input, int N, T* output, int coarse_factor){
    unsigned int segment = coarse_factor*2*blockDim.x*blockIdx.x;
    unsigned int i = segment + threadIdx.x;

    // Load data to shared memory
    extern __shared__ float input_s[];
    float threadSum = 0.0f;
    for (unsigned int c = 0; c < coarse_factor*2; ++c) {
        threadSum += input[i + c*blockDim.x];
    }
    input_s[threadIdx.x] = threadSum;

    __syncthreads();

    // Reduction tree in shared memory
    for(unsigned int stride = blockDim.x/2; stride > 0; stride /= 2) {
        input_s[threadIdx.x] += input_s[threadIdx.x + stride];
    }
    __syncthreads();

    if(threadIdx.x == 0){
        atomicAdd(output, input_s[0]);
    }
}

/****************Functions************************/

template <typename T>
void sum_coarsened(T* input, int N, T* output){
    int coarse_factor = 2; 
    int blockSize = 1024;
    if (N < 1024 * 4) {
        blockSize = N/4;
    }
    int gridSize = ((N/2) + (blockSize * coarse_factor) - 1) / (blockSize * coarse_factor);
    dim3 blockDim(blockSize);
    dim3 gridDim(gridSize);

    sum_coarsened_kernel<<<gridDim, blockDim, blockSize>>>(input, N, output, coarse_factor);
    checkCudaErrors(cudaGetLastError()); 
}

template <typename T>
void sum_shared_mem(T* input, int N, T* output){
    int blockSize = 1024;
    if (N < 2048) {
        blockSize = N/2;
    }
    int gridSize = ((N/2) + (blockSize - 1)) / blockSize;
    dim3 blockDim(blockSize);
    dim3 gridDim(gridSize);

    sum_shared_mem_kernel<<<gridDim, blockDim, blockSize>>>(input, N, output);
    checkCudaErrors(cudaGetLastError());
}

template <typename T>
void sum_seg_reduction(T* input, int N, T* initial){
    int blockSize = 1024;
    if (N < 2048) {
        blockSize = N/2;
    }
    int gridSize = ((N/2) + (blockSize - 1)) / blockSize;
    dim3 blockDim(blockSize);
    dim3 gridDim(gridSize);
    
    sum_seg_reduction_kernel<<<gridDim, blockDim>>>(input, N, initial);
    checkCudaErrors(cudaGetLastError());
}

template <typename T>
void sum(T* input, int N, T* initial) {
    sum_kernel<<<1,1>>>(input, N, initial);
    checkCudaErrors(cudaGetLastError());
}

template <typename T>
void sum_atomic(T* input, int N, T* initial) {
    int blockSize = 1024;
    if (N < 2048) {
        blockSize = N/2;
    }
    int gridSize = (N + blockSize - 1) / blockSize;
    dim3 blockDim(blockSize);
    dim3 gridDim(gridSize);

    sum_atomic_kernel<<<gridDim, blockDim>>>(input, N, initial);
    checkCudaErrors(cudaGetLastError());
}

template <typename T>
void sum_coalecsing(T* input, int N, T* output){
    int blockSize = 1024;
    if (N < 2048) {
        blockSize = N/2;
    }
    int gridSize = ((N/2) + (blockSize - 1)) / blockSize;
    dim3 blockDim(blockSize);
    dim3 gridDim(gridSize);
     
    sum_coalecsing_kernel<<<gridDim, blockDim>>>(input, N, output);
    checkCudaErrors(cudaGetLastError());
}

template void sum_coarsened<float>(float*, int, float*);
template void __global__ sum_coarsened_kernel<float>(float*, int, float*, int);

template void sum_shared_mem<float>(float*, int, float*);
template __global__ void sum_shared_mem_kernel<float>(float*, int, float*);

template void sum_coalecsing<float>(float*, int, float*);
template __global__ void sum_coalecsing_kernel<float>(float*, int, float*);

template void sum<float> (float*, int, float*);
template __global__ void sum_kernel<float> (float*, int, float*);

template void sum_atomic<float> (float*, int, float*);
template __global__ void sum_atomic_kernel<float> (float*, int, float*);

template __global__ void sum_seg_reduction_kernel<float> (float*, int, float*);
template void sum_seg_reduction<float> (float*, int, float*);
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
