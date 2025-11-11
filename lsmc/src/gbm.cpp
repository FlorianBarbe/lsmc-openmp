/**
 * Implémentation du module GBM (Geometric Brownian Motion)
 */

#include "gbm.hpp"
#include <cmath>

GBM::GBM(double S0, double r, double sigma, double T, int N_steps)
    : S0(S0), r(r), sigma(sigma), T(T), N_steps(N_steps) {
}

std::vector<double> GBM::simulate(RNG& rng) {
    std::vector<double> path(N_steps + 1);
    path[0] = S0;
    double dt = T / N_steps;

    for (int i = 1; i <= N_steps; ++i) {
        double Z = rng.normal();
        path[i] = path[i - 1] * std::exp((r - 0.5 * sigma * sigma) * dt
            + sigma * std::sqrt(dt) * Z);
    }

    return path;
}
