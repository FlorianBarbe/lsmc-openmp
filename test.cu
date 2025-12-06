#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "gbm.hpp"

int main() {
    GbmParams params{ 100.0f, 100.0f, 0.05f, 0.2f, 1.0f, 50, 200000 };
    size_t total = (params.nSteps + 1) * params.nPaths;
    float* d_paths = nullptr;
    cudaMalloc(&d_paths, total * sizeof(float));

    simulate_gbm_paths_cuda(params, RNGType::PseudoPhilox, d_paths);

    std::cout << "GPU simulation done!" << std::endl;
    cudaFree(d_paths);

    return 0;
}
