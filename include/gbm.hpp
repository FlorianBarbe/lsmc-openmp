#pragma once
#include <vector>
#include <string>
#include "rng.hpp"
#include <cuda_runtime.h>   // pour cudaStream_t


class GBM {
private:
    double S0;
    double K;
    double r;
    double sigma;
    double T;
    int N_steps;

public:
    GBM(double S0, double r, double sigma, double T, int N_steps);

    void simulate_path(RNG& rng, double* path_out);

    static void simulatePaths(double* paths, double S0, double r, double sigma,
        double T, int N_steps, int N_paths);

    static void exportCSV(const std::vector<std::vector<double>>& paths,
        const std::string& filename);
};

#pragma once


// Paramètres du GBM utilisés côté GPU
struct GbmParams {
    double S0;
    double K;
    double r;
    double sigma;
    double T;
    int   nSteps;
    int   nPaths;
};

// Type de RNG utilisé par la version CUDA
enum class RNGType {
    PseudoPhilox,
    SobolScrambled
};

// API host : déclaration unique (pas de redéfinition dans le .cu)
void simulate_gbm_paths_cuda(const GbmParams& params,
    RNGType rng,
    float* d_paths,
    unsigned long long seed = 1234ULL,
    cudaStream_t stream = 0);