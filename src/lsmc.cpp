/**
 * Implémentation du module Longstaff–Schwartz (parallélisée avec OpenMP)
 */

#include "lsmc.hpp"
#include <cmath>
#include <algorithm>
#include <omp.h>

double LSMC::priceAmericanPut(double S0, double K, double r, double sigma,
    double T, int N_steps, int N_paths) {
    double dt = T / N_steps;
    double discount = std::exp(-r * dt);
    std::vector<std::vector<double>> paths(N_paths);

    GBM gbm(S0, r, sigma, T, N_steps);

    // ==========================================
    // 1. Simulation des trajectoires (parallèle)
    // ==========================================
#pragma omp parallel
    {
        RNG rng; // générateur propre à chaque thread
#pragma omp for schedule(static)
        for (int i = 0; i < N_paths; ++i)
            paths[i] = gbm.simulate(rng);
    }

    // ==========================================
    // 2. Calcul des payoffs
    // ==========================================
    std::vector<std::vector<double>> payoff(N_paths, std::vector<double>(N_steps + 1, 0.0));

#pragma omp parallel for schedule(static)
    for (int i = 0; i < N_paths; ++i)
        for (int t = 0; t <= N_steps; ++t)
            payoff[i][t] = std::max(K - paths[i][t], 0.0);

    // ==========================================
    // 3. Backward induction (séquentiel)
    // ==========================================
    std::vector<double> cashflows(N_paths);
    for (int i = 0; i < N_paths; ++i)
        cashflows[i] = payoff[i][N_steps];

    for (int t = N_steps - 1; t > 0; --t) {
        std::vector<double> X, Y;
        for (int i = 0; i < N_paths; ++i)
            if (payoff[i][t] > 0.0) {
                X.push_back(paths[i][t]);
                Y.push_back(cashflows[i] * discount);
            }

        if (X.empty()) continue;

        auto coeffs = Regression::ols(X, Y);

#pragma omp parallel for schedule(static)
        for (int i = 0; i < N_paths; ++i) {
            if (payoff[i][t] > 0.0) {
                double continuation = coeffs[0] + coeffs[1] * paths[i][t] + coeffs[2] * paths[i][t] * paths[i][t];
                if (payoff[i][t] > continuation)
                    cashflows[i] = payoff[i][t];
                else
                    cashflows[i] *= discount;
            }
            else {
                cashflows[i] *= discount;
            }
        }
    }

    // ==========================================
    // 4. Moyenne finale
    // ==========================================
    double mean = 0.0;
#pragma omp parallel for reduction(+:mean)
    for (int i = 0; i < N_paths; ++i)
        mean += cashflows[i];

    return mean / N_paths;
}
