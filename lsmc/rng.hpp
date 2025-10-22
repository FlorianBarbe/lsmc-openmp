/**
 * rng.hpp
 * Génération de nombres aléatoires (loi normale standard)
 */
#pragma once
#include <random>

class RNG {
private:
    std::mt19937 gen;
    std::normal_distribution<> dist;

public:
    RNG() : gen(std::random_device{}()), dist(0.0, 1.0) {}
    double normal() { return dist(gen); }
};
