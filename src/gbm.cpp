#include "gbm.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>

GBM::GBM(double S0, double r, double sigma, double T, int N_steps)
    : S0(S0), r(r), sigma(sigma), T(T), N_steps(N_steps) {
}

// -------------------------------------------
// Simule UNE trajectoire (méthode membre)
// -------------------------------------------
std::vector<double> GBM::simulate(RNG& rng) {
    std::vector<double> path(N_steps + 1);
    double dt = T / N_steps;

    path[0] = S0;

    for (int i = 1; i <= N_steps; ++i) {
        double Z = rng.normal();
        path[i] = path[i - 1] * std::exp((r - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * Z);
    }

    return path;
}

// -------------------------------------------
// Simule PLUSIEURS trajectoires (méthode statique)
// -------------------------------------------
std::vector<std::vector<double>> GBM::simulatePaths(
    double S0, double r, double sigma, double T, int N_steps, int N_paths)
{
    std::vector<std::vector<double>> paths(N_paths);
    double dt = T / N_steps;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < N_paths; ++i) {
        RNG rng(i + 1234);
        GBM model(S0, r, sigma, T, N_steps);

        paths[i] = model.simulate(rng);
    }

    return paths;
}

// -------------------------------------------
// Export CSV
// -------------------------------------------
void GBM::exportCSV(
    const std::vector<std::vector<double>>& paths,
    const std::string& filename)
{
    std::ofstream f(filename);

    if (!f.is_open()) return;

    for (const auto& p : paths) {
        for (size_t i = 0; i < p.size(); ++i) {
            f << std::fixed << std::setprecision(6) << p[i];
            if (i + 1 < p.size()) f << ",";
        }
        f << "\n";
    }

    f.close();
}
