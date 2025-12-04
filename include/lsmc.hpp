/**
 * Projet LSMC – Module Longstaff–Schwartz
 * ---------------------------------------
 * Rôle : encapsule la logique du pricing d’une option américaine via Monte Carlo + régression.
 */

#pragma once
#include <vector>
#include "gbm.hpp"
#include "regression.hpp"
#include "rng.hpp"
#include <cmath>
#include <algorithm>
#include <omp.h>

#include <cuda_runtime.h>


class LSMC {
public:
    static double priceAmericanPut(double S0, double K, double r, double sigma,
        double T, int N_steps, int N_paths);
    double priceAmericanPutGPU(double S0, double K, double r, double sigma,
        double T, int N_steps, int N_paths);
};
// API host : déclaration unique (pas de redéfinition dans le .cu)
void simulate_gbm_paths_cuda(const GbmParams& params,
    RNGType rng,
    float* d_paths,
    unsigned long long seed = 1234ULL,
    cudaStream_t stream = 0);

// Structure pour accumuler les sommes nécessaires à la régression sur GPU
struct RegressionSumsGPU {
    double a00, a01, a02;
    double a11, a12, a22;
    double b0, b1, b2;
};

// Fonction pour calculer les sommes de régression sur GPU
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
