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

#ifdef _OPENMP
#include <omp.h>
#endif



class LSMC {
public:
    static double priceAmericanPut(double S0, double K, double r, double sigma,
    double T, int N_steps, int N_paths);
};
