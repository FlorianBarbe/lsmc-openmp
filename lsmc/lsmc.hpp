/**
 * lsmc.hpp
 * Implémentation de la méthode de Longstaff–Schwartz (LSMC)
 */
#pragma once
#include <vector>

namespace LSMC {
    double priceAmericanPut(double S0, double K, double r, double sigma,
        double T, int N_steps, int N_paths);
}
#pragma once
