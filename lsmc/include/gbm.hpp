/**
 * Projet LSMC – Module GBM (Geometric Brownian Motion)
 * ----------------------------------------------------
 * Rôle : simuler une trajectoire de prix selon le modèle de mouvement brownien géométrique.
 */

#pragma once
#include <vector>
#include "rng.hpp"

class GBM {
public:
    GBM(double S0, double r, double sigma, double T, int N_steps);
    std::vector<double> simulate(RNG& rng);

private:
    double S0;     // prix initial
    double r;      // taux sans risque
    double sigma;  // volatilité
    double T;      // horizon temporel
    int N_steps;   // nombre de pas
};
