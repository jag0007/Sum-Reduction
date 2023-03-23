#include <random>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <sumReduce.h>

std::vector<int> parseInputParams(int, char**);
template <typename T> std::vector<T> getListOfRandomValues(int);
template <typename T> T sumReduce(T*, int, T);
template <typename T> void printVector(std::vector<T>, const char*);

int main(int argc, char *argv[]) {

    std::vector<int> inputParams = parseInputParams(argc, argv);
    //int listLength = getNumberOfValues(argc, argv);
    auto values = getListOfRandomValues<float>(inputParams[1]);
    printVector(values, "Input Params");
   
    auto sum = std::accumulate(values.begin(), values.end(), decltype(values)::value_type(0));
 
    for (auto& value : values)
        std::cout << value << " ";
    std::cout << std::endl;
    std::cout << "sum: " << sum << std::endl;

    //auto gpuSum = sumReduce(values.data(), values.size(), 0.0f);
    auto gpuSum = sumReduce(values.data(), values.size(), 0.0f);

    std::cout << "gpuSum: " << gpuSum << std::endl;

}

template <typename T>
T sumReduce(T* inputValues, int N, T initialValue) {
    float * input = nullptr;
    float * initial = nullptr;
    size_t inputSize = N * sizeof(T);
    size_t initialSize = sizeof(T);

    // allocate memory on GPU
    checkCudaErrors(cudaMalloc(&input, inputSize));
    checkCudaErrors(cudaMalloc(&initial, initialSize));

    // load it up
    checkCudaErrors(cudaMemcpy(input, inputValues, inputSize, cudaMemcpyHostToDevice));
    //checkCudaErrors(cudaMemcpy(initial, &initialValue, initialSize, cudaMemcpyHostToDevice));

    //sum<T>(input, N, initial);
    //sum_atomic<T>(input, N, initial);
    sum_seg_reduction<T>(input, N);

    // read back result
    T result;
    //checkCudaErrors(cudaMemcpy(&result, initial, sizeof(T), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&result, input, sizeof(T), cudaMemcpyDeviceToHost));

    // free
    checkCudaErrors(cudaFree(input));

    return result;
}

template <typename T>
void printVector(std::vector<T> input, const char* name) {
    std::cout << name << ":" << std::endl;
    for (auto& i : input) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

std::vector<int> parseInputParams(int argc, char *argv[]) {
    if (argc != 3){
        throw std::runtime_error("Incorrect number of input arguments");
    }

    int printArrayValues = std::stoi(argv[1]);
    int getNumberOfValues = std::stoi(argv[2]);
    return std::vector<int> {printArrayValues, getNumberOfValues};
}


template <typename T>
std::vector<T> getListOfRandomValues(int listLength) {
    std::vector<T> output;
    output.reserve(listLength);


    std::uniform_real_distribution<T> unif(-1000, 1000);
    std::default_random_engine re;
    for (int i = 0; i < listLength; i++) {
        output.push_back(unif(re));
    }


    return output;
}