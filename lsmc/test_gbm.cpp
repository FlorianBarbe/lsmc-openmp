#include <iostream>
#include <cmath>
#include "gbm.hpp"
#include "rng.hpp"

int main() {
    double S0 = 100.0, r = 0.05, sigma = 0.2, T = 1.0;
    int N_steps = 50, N_paths = 10000;

    GBM gbm(S0, r, sigma, T, N_steps);
    double sum = 0.0;

    for (int i = 0; i < N_paths; ++i) {
        RNG rng;
        auto path = gbm.simulate(rng);
        sum += path.back();
    }

    double mean = sum / N_paths;
    double expected = S0 * std::exp(r * T);
    double rel_error = std::abs(mean - expected) / expected;

    std::cout << "=== Test du module GBM ===" << std::endl;
    std::cout << "Moyenne simulée     : " << mean << std::endl;
    std::cout << "Valeur théorique    : " << expected << std::endl;
    std::cout << "Erreur relative (%) : " << rel_error * 100 << std::endl;

    if (rel_error < 2.0 / 100.0)
        std::cout << "[OK] Test réussi (cohérent avec la théorie)" << std::endl;
    else
        std::cout << "[ERREUR] Test échoué (écart trop grand)" << std::endl;
}
#pragma once
