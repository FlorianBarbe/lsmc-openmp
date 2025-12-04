#include "gbm.hpp"
#include "lsmc.hpp"
#include <iostream>
#include <vector>

int main()
{
    GbmParams p;
    p.S0 = 100;
    p.K = 100.0;
    p.r = 0.02;
    p.sigma = 0.2;
    p.T = 1.0;
    p.nSteps = 10;
    p.nPaths = 4;

    const int total = (p.nSteps + 1) * p.nPaths;

    float* d_paths;
    cudaMalloc(&d_paths, total * sizeof(float));

    simulate_gbm_paths_cuda(p, RNGType::PseudoPhilox, d_paths);

    std::vector<float> h(total);
    cudaMemcpy(h.data(), d_paths, total * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_paths);


    LSMC engine;
    double price_gpu = engine.priceAmericanPutGPU(p.S0, p.K, p.r, p.sigma, p.T,p.nSteps, p.nPaths);

        std::cout << "American put (GPU paths) = " << price_gpu << std::endl;

    std::cout << "GPU OK, paths:\n";
    for (int i = 0; i < total; i++) std::cout << h[i] << " ";
    return 0;
}
