/**
 * Projet LSMC – Module RNG (Random Number Generator)
 * --------------------------------------------------
 * Rôle : générer des nombres aléatoires suivant une loi normale standard.
 */

#pragma once
#include <random>

class RNG {
public:
    RNG() : gen(std::random_device{}()), dist(0.0, 1.0) {}

    // Retourne un échantillon de loi normale N(0,1)
    double normal() {
        return dist(gen);
    }

private:
    std::mt19937 gen; // générateur Mersenne Twister
    std::normal_distribution<double> dist;
};
