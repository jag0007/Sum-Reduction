#include <random>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <numeric>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <sumReduce.h>
#include <functional>

std::vector<int> parseInputParams(int, char**);
template <typename T> std::vector<T> getListOfRandomValues(int);
float sumReduce(float*, int, float, sumType);
template <typename T> void printVector(std::vector<T>, const char*);

int main(int argc, char *argv[]) {

    std::vector<int> inputParams = parseInputParams(argc, argv);
    //int listLength = getNumberOfValues(argc, argv);
    auto values = getListOfRandomValues<float>(inputParams[1]);
    if (inputParams[0] == 0)
        printVector(values, "Input Params");
   
    auto sum = std::accumulate(values.begin(), values.end(), decltype(values)::value_type(0));
 
    for (auto& value : values)
        std::cout << value << " ";
    std::cout << std::endl;
    std::cout << "sum: " << sum << std::endl;

    //auto gpuSum = sumReduce(values.data(), values.size(), 0.0f);
    std::vector<sumType> sumTypes = { sumType::SUM, sumType::ATOMIC, sumType::SEG, sumType::COAL, sumType::SHARED, sumType::COARSE };
    for (sumType type : sumTypes) {
        auto gpuSum = sumReduce(values.data(), values.size(), 0.0f, type);
        std::cout << "gpuSum: " << gpuSum << std::endl;
    }
}

float sumReduce(float* inputValues, int N, float initialValue, sumType sum_type) {
    float * input = nullptr;
    size_t inputSize = N * sizeof(float);
    checkCudaErrors(cudaMalloc(&input, inputSize));
    checkCudaErrors(cudaMemcpy(input, inputValues, inputSize, cudaMemcpyHostToDevice));

    float * initial = nullptr;
    size_t initialSize = sizeof(float);
    checkCudaErrors(cudaMalloc(&initial, initialSize));
    checkCudaErrors(cudaMemcpy(initial, &initialValue, initialSize, cudaMemcpyHostToDevice));

    void (*sum_func) (float*, int, float*);
    switch (sum_type) {
        case sumType::SUM:
            sum_func = &sum<float>;
            break;
        case sumType::ATOMIC:
            sum_func = &sum_atomic<float>;
            break;
        case sumType::SEG:
            sum_func = &sum_seg_reduction<float>;
            break;
        case sumType::COAL:
            sum_func = &sum_coalecsing<float>;
            break;
        case sumType::SHARED:
            sum_func = &sum_shared_mem<float>;
            break;
        case sumType::COARSE:
            sum_func = &sum_coarsened<float>;
            break;
    }

    // for timer besting
    sum_func(input, N, initial);

    // read back result
    float result;
    if (sum_type != sumType::SUM){
        checkCudaErrors(cudaMemcpy(&result, initial, sizeof(float), cudaMemcpyDeviceToHost));
    } else {
        checkCudaErrors(cudaMemcpy(&result, input, sizeof(float), cudaMemcpyDeviceToHost));
    }

    // free
    checkCudaErrors(cudaFree(input));
    checkCudaErrors(cudaFree(initial));

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