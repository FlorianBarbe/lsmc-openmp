#pragma once
#include <vector>
#include <string>
#include "rng.hpp"

class GBM {
public:
    // Constructeur : paramètres du modèle
    GBM(double S0, double r, double sigma, double T, int N_steps);

    // Simule UNE trajectoire
    std::vector<double> simulate(RNG& rng);

    // Simule PLUSIEURS trajectoires
    static std::vector<std::vector<double>> simulatePaths(
        double S0, double r, double sigma, double T, int N_steps, int N_paths
    );

    // Exporte un tableau de trajectoires dans un CSV
    static void exportCSV(
        const std::vector<std::vector<double>>& paths,
        const std::string& filename
    );

private:
    double S0;
    double r;
    double sigma;
    double T;
    int N_steps;
};
