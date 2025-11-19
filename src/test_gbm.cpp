/**
 * Test unitaire : validation du module GBM
 * ----------------------------------------
 * Objectif : vérifier que la moyenne empirique du prix final S_T
 * sur un grand nombre de trajectoires correspond à l'espérance théorique
 * du mouvement brownien géométrique :
 *      E[S_T] = S0 * exp(r * T)
 */

#include <iostream>
#include <cmath>
#include "gbm.hpp"
#include "rng.hpp"

int main() {
    // Paramètres du test
    double S0 = 100.0;
    double r = 0.05;
    double sigma = 0.2;
    double T = 1.0;
    int N_steps = 50;
    int N_paths = 10000;

    GBM gbm(S0, r, sigma, T, N_steps);
    double sum = 0.0;

    // Simulation de N trajectoires
    for (int i = 0; i < N_paths; ++i) {
        RNG rng;
        auto path = gbm.simulate(rng);
        sum += path.back(); // prix final
    }

    // Moyenne empirique et valeur théorique
    double mean = sum / N_paths;
    double expected = S0 * std::exp(r * T);
    double rel_error = std::abs(mean - expected) / expected;

    std::cout << "=== Test du module GBM ===" << std::endl;
    std::cout << "Moyenne simulée     : " << mean << std::endl;
    std::cout << "Valeur théorique    : " << expected << std::endl;
    std::cout << "Erreur relative (%) : " << rel_error * 100 << std::endl;

    if (rel_error < 2.0 / 100.0)
        std::cout << "Résultat : ✅ Test réussi (cohérent avec la théorie)" << std::endl;
    else
        std::cout << "Résultat : ❌ Test échoué (écart trop grand)" << std::endl;

    return 0;
}
