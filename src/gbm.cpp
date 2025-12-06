#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <cmath>

#include "gbm.hpp"

using namespace std;


using namespace std;

GBM::GBM(double S0, double r, double sigma, double T, int N_steps)
    : S0(S0), r(r), sigma(sigma), T(T), N_steps(N_steps) {
}

// Simulation complète et retour d'un vecteur pour les tests
std::vector<double> GBM::simulate(RNG& rng) {
    std::vector<double> path(static_cast<size_t>(N_steps) + 1);
    simulate_path(rng, path.data());
    return path;
}


// Simulation d'une trajectoire
void GBM::simulate_path(RNG& rng, double* path_out) {
    double dt = T / N_steps;

    path_out[0] = S0;
    double S = S0;

    for (int t = 1; t <= N_steps; t++) 
    {
        double Z = rng.normal();
		S = S * exp((r - 0.5 * sigma * sigma) * dt + sigma * sqrt(dt) * Z);
        path_out[t] = S;
    }
}

// Simulation de plusieurs trajectoires
void GBM::simulatePaths(double* paths, double S0, double r, double sigma, double T, int N_steps, int N_paths)
{
#pragma omp parallel for schedule(static)
    for (int i = 0; i < N_paths; ++i) {
        RNG rng(i + 1234);
        GBM model(S0, r, sigma, T, N_steps);
        double* path_ptr = paths + i * (N_steps + 1);
		model.simulate_path(rng, path_ptr);
    }
}

// Export de trajectoires_gbm.csv

void GBM::exportCSV(
    const std::vector<std::vector<double>>& paths,
    const std::string& filename)
{
    const std::string tmpname = filename + ".tmp";

    // 1) Écriture dans un fichier temporaire
    std::ofstream f(tmpname, std::ios::trunc);
    if (!f.is_open()) {
        std::cerr << "[ERREUR] Impossible d'ouvrir le fichier temporaire : "
            << tmpname << "\n";
        return;
    }

    const size_t cols = paths[0].size();

    for (size_t p = 0; p < paths.size(); ++p) {

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

    // 2) Remplacement atomique : écrase l'ancien d'un seul coup
    try {
        std::filesystem::rename(tmpname, filename);
    }
    catch (std::filesystem::filesystem_error& e) {
        // Si le rename échoue (Windows si fichier verrouillé), on supprime l'ancien puis on réessaie
        std::cerr << "[INFO] Rename impossible, tentative de suppression : "
            << e.what() << "\n";

        std::error_code ec;
        std::filesystem::remove(filename, ec);   // supprime l'ancien sans arrêter le programme
        std::filesystem::rename(tmpname, filename);
    }
}