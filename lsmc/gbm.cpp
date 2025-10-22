#include "gbm.hpp"
#include <cmath>
#include <fstream>
#ifdef _OPENMP
#include <omp.h>
#endif

std::vector<double> GBM::simulate(RNG& rng) {
    std::vector<double> path(N_steps + 1);
    path[0] = S0;
    double dt = T / N_steps;
    for (int i = 1; i <= N_steps; ++i) {
        double z = rng.normal();
        path[i] = path[i - 1] * std::exp((r - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * z);
    }
    return path;
}

// === AJOUTS ===
std::vector<std::vector<double>>
GBM::simulatePaths(double S0, double r, double sigma, double T, int N_steps, int N_paths) {
    std::vector<std::vector<double>> paths(N_paths);
#pragma omp parallel for
    for (int i = 0; i < N_paths; ++i) {
        RNG local_rng;
        GBM gbm(S0, r, sigma, T, N_steps);
        paths[i] = gbm.simulate(local_rng);
    }
    return paths;
}

void GBM::exportCSV(const std::vector<std::vector<double>>& paths,
    const std::string& filename) {
    std::ofstream f(filename);
    if (!f) return;
    for (const auto& p : paths) {
        for (size_t j = 0; j < p.size(); ++j) {
            f << p[j];
            if (j + 1 < p.size()) f << ",";
        }
        f << "\n";
    }
}
