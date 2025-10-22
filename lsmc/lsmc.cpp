#include "lsmc.hpp"
#include "gbm.hpp"
#include "rng.hpp"
#include "regression.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

double LSMC::priceAmericanPut(double S0, double K, double r, double sigma,
    double T, int N_steps, int N_paths) {
    double dt = T / N_steps;
    std::vector<std::vector<double>> paths(N_paths);
    RNG rng;

    // Simulation Monte Carlo
#pragma omp parallel for
    for (int i = 0; i < N_paths; ++i) {
        RNG local_rng;
        GBM gbm(S0, r, sigma, T, N_steps);
        paths[i] = gbm.simulate(local_rng);
    }

    std::vector<std::vector<double>> cashflows(N_paths, std::vector<double>(N_steps + 1, 0.0));

    // Payoff final
    for (int i = 0; i < N_paths; ++i)
        cashflows[i][N_steps] = std::max(K - paths[i][N_steps], 0.0);

    // Backward induction
    for (int t = N_steps - 1; t >= 1; --t) {
        std::vector<double> X, Y;
        for (int i = 0; i < N_paths; ++i) {
            double payoff = std::max(K - paths[i][t], 0.0);
            if (payoff > 0.0) {
                X.push_back(paths[i][t]);
                double discounted = cashflows[i][t + 1] * std::exp(-r * dt);
                Y.push_back(discounted);
            }
        }

        if (X.size() < 3) continue;

        auto coef = Regression::ols_poly2(X, Y);
        auto continuation = Regression::predict_poly2(X, coef);

        int idx = 0;
        for (int i = 0; i < N_paths; ++i) {
            double payoff = std::max(K - paths[i][t], 0.0);
            if (payoff > 0.0) {
                if (payoff > continuation[idx])
                    cashflows[i][t] = payoff;
                else
                    cashflows[i][t] = cashflows[i][t + 1] * std::exp(-r * dt);
                idx++;
            }
            else {
                cashflows[i][t] = cashflows[i][t + 1] * std::exp(-r * dt);
            }
        }
    }

    // Moyenne actualisée
    double sum = 0.0;
    for (int i = 0; i < N_paths; ++i) {
        for (int t = 0; t <= N_steps; ++t)
            if (cashflows[i][t] > 0.0) {
                sum += cashflows[i][t] * std::exp(-r * t * dt);
                break;
            }
    }

    return sum / N_paths;
}
