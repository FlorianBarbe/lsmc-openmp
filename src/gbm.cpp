#include "gbm.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>

GBM::GBM(double S0, double r, double sigma, double T, int N_steps)
    : S0(S0), r(r), sigma(sigma), T(T), N_steps(N_steps) {
}

// Simulation d'une trajectoire
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

// Simulation de plusieurs trajectoires
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

// Export de trajectoires_gbm.csv
void GBM::exportCSV(
    const std::vector<std::vector<double>>& paths,
    const std::string& filename)
{
    std::ofstream f(filename, std::ios::trunc);

    if (!f.is_open()) {
        std::cerr << "[ERREUR] Impossible d'ouvrir le fichier : "
            << filename << "\n";
        return;
    }

    const size_t cols = paths[0].size();

    for (size_t p = 0; p < paths.size(); ++p) {

        // sécurité : ligne anormale → on saute
        if (paths[p].size() != cols) {
            std::cerr << "[AVERTISSEMENT] Ligne "
                << p
                << " ignorée (taille incorrecte)\n";
            continue;
        }

        for (size_t i = 0; i < cols; ++i) {
            f << std::fixed << std::setprecision(6) << paths[p][i];
            if (i + 1 < cols) f << ",";
        }
        f << "\n";
    }

    f.flush();
    f.close();
}

