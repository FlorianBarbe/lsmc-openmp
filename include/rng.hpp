/**
 * Projet LSMC – Module RNG (Random Number Generator)
 * --------------------------------------------------
 * Rôle : générer des nombres aléatoires suivant une loi normale standard.
 */

#pragma once
#include <random>

class RNG {
public:
    // Nouveau : constructeur avec seed (OBLIGATOIRE pour simulatePaths)
    RNG(int seed = std::random_device{}())
        : gen(seed), dist(0.0, 1.0) {
    }

    // Retourne un échantillon de loi normale N(0,1)
    double normal() {
        return dist(gen);
    }

private:
    std::mt19937 gen; // générateur Mersenne Twister
    std::normal_distribution<double> dist;
};
