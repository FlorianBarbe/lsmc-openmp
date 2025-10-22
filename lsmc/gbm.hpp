#pragma once
#include <vector>
#include <string>
#include "rng.hpp"

class GBM {
private:
    double S0, r, sigma, T;
    int N_steps;

public:
    GBM(double S0, double r, double sigma, double T, int N_steps)
        : S0(S0), r(r), sigma(sigma), T(T), N_steps(N_steps) {
    }

    // Simulation d’une trajectoire unique
    std::vector<double> simulate(RNG& rng);

    // Simulation de plusieurs trajectoires
    static std::vector<std::vector<double>>
        simulatePaths(double S0, double r, double sigma, double T, int N_steps, int N_paths);

    // Export CSV des trajectoires
    static void exportCSV(const std::vector<std::vector<double>>& paths,
        const std::string& filename);
};
