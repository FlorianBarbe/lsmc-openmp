/**
 * Projet LSMC – Module Longstaff–Schwartz
 * ---------------------------------------
 * Rôle : encapsule la logique du pricing d’une option américaine via Monte Carlo + régression.
 */

#pragma once

#pragma once
#ifndef LSMC_ENABLE_CUDA
#ifdef __CUDACC__
#define LSMC_ENABLE_CUDA 1
#endif
#endif

#include <vector>
#include "gbm.hpp"
#include "regression.hpp"
#include "rng.hpp"
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef LSMC_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

class LSMC {
public:
    static double priceAmericanPut(double S0, double K, double r, double sigma,
    double T, int N_steps, int N_paths);
    #ifdef LSMC_ENABLE_CUDA
    double priceAmericanPutGPU(double S0, double K, double r, double sigma,
        double T, int N_steps, int N_paths);
    #endif
};
// API host : déclaration unique (pas de redéfinition dans le .cu)
#ifdef LSMC_ENABLE_CUDA
void simulate_gbm_paths_cuda(const GbmParams& params,
    RNGType rng,
    float* d_paths,
    unsigned long long seed ,
    cudaStream_t stream );
#endif

// Structure pour accumuler les sommes nécessaires à la régression sur GPU
struct RegressionSumsGPU {
    double a00, a01, a02;
    double a11, a12, a22;
    double b0, b1, b2;
};

// Fonction pour calculer les sommes de régression sur GPU
#ifdef LSMC_ENABLE_CUDA
void computeRegressionSumsGPU(const float* d_paths,
    const float* d_payoff,
    const float* d_cashflows,
    int t,
    int N_steps,
    int N_paths,
    float discount,
    RegressionSumsGPU& out);

// Structure pour stocker les coefficients de la régression bêta
struct BetaGPU {
    double beta0, beta1, beta2;
};

// Fonction pour mettre à jour les cashflows sur GPU
void updateCashflowsGPU(const float* d_paths,
    const float* d_payoff,
    float* d_cashflows,
    const BetaGPU& beta,
    float discount,
    int t,
    int N_steps,
    int N_paths);
#endif
