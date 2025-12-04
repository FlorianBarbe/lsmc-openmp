#pragma once

#include <cuda_runtime.h>

// =========================================================
// PARAMÈTRES GBM POUR CUDA — doivent matcher gbm.cu EXACTEMENT
// =========================================================
struct GbmParams {
    float S0;
    float r;
    float sigma;
    float T;
    int   nSteps;
    int   nPaths;
};

// Types de RNG disponibles pour CUDA
enum class RNGType {
    PseudoPhilox,
    SobolScrambled
};

// =========================================================
// Déclaration du simulateur GPU
// =========================================================

void simulate_gbm_paths_cuda(const GbmParams& params,
    RNGType rng,
    float* d_paths,
    unsigned long long seed = 1234ULL,
    cudaStream_t stream = 0);


// =========================================================
// PARTIE CPU (classe GBM)
// =========================================================

#include <vector>
#include <string>
#include "rng.hpp"

class GBM {
private:
    double S0;
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
