#pragma once


// =========================================================
// PARTIE CPU (classe GBM)
// =========================================================

#include <vector>
#include <string>
#include "rng.hpp"

class GBM {
private:
    double S0;
    double r;
    double sigma;
    double T;
    int N_steps;

public:
    GBM(double S0, double r, double sigma, double T, int N_steps);

    void simulate_path(RNG& rng, double* path_out);
    std::vector<double> simulate(RNG& rng);
    static void simulatePaths(double* paths, double S0, double r, double sigma,
        double T, int N_steps, int N_paths, int seed);

    static void exportCSV(const std::vector<std::vector<double>>& paths,
        const std::string& filename);
};
