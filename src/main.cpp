/**
 * main.cpp — version clean pour Streamlit + CSV multi-threads
 * -----------------------------------------------------------
 * Usage :
 *   lsmc.exe S0 K r sigma T N_steps N_paths
 */

#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>

#include "lsmc.hpp"
#include "gbm.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char** argv) {

    if (argc < 8) {
        std::cerr << "Usage : lsmc.exe S0 K r sigma T N_steps N_paths\n";
        return 1;
    }

    // =============================
    // LECTURE DES PARAMÈTRES
    // =============================
    double S0 = std::stod(argv[1]);
    double K = std::stod(argv[2]);
    double r = std::stod(argv[3]);
    double sigma = std::stod(argv[4]);
    double T = std::stod(argv[5]);
    int N_steps = std::stoi(argv[6]);
    int N_paths = std::stoi(argv[7]);

    std::cout << std::fixed << std::setprecision(6);

#ifdef _OPENMP
    std::cout << "[INFO] OpenMP activé ("
        << omp_get_max_threads()
        << " threads disponibles)\n";
#else
    std::cout << "[INFO] OpenMP non activé — mode séquentiel\n";
#endif

    std::cout << "[INFO] Simulation de "
        << N_paths << " trajectoires, "
        << N_steps << " pas temporels\n";

    // =============================
    // SIMULATION GBM
    // =============================
    auto paths = GBM::simulatePaths(S0, r, sigma, T, N_steps, N_paths);
    GBM::exportCSV(paths, "trajectoires_gbm.csv");


    // =============================
    // CRÉATION/OUVERTURE DU CSV
    // =============================
    std::string csv_file = "resultats_lsmc.csv";
    bool file_exists = std::filesystem::exists(csv_file);

    std::ofstream f(csv_file, std::ios::app);

    // ---------- En-tête (une seule fois) ----------
    if (!file_exists) {
        f << "Threads,S0,K,r,sigma,T,N_steps,N_paths,Prix,Temps\n";
    }

    // =============================
    // EXÉCUTION SÉQUENTIELLE
    // =============================
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        double price = LSMC::priceAmericanPut(S0, K, r, sigma, T, N_steps, N_paths);
        auto t2 = std::chrono::high_resolution_clock::now();

        double dt = std::chrono::duration<double>(t2 - t1).count();

        f << 1 << "," << S0 << "," << K << "," << r << "," << sigma << ","
            << T << "," << N_steps << "," << N_paths << ","
            << price << "," << dt << "\n";

        std::cout << "[SEQ] Temps = " << dt
            << " | Prix = " << price << "\n";
    }


    // =============================
    // OPENMP : tests threads = 2,4,8,16
    // =============================
#ifdef _OPENMP
    std::vector<int> thread_list = { 2, 4, 8, 16 };

    for (int th : thread_list) {
        omp_set_num_threads(th);

        auto t1 = std::chrono::high_resolution_clock::now();
        double price = LSMC::priceAmericanPut(S0, K, r, sigma, T, N_steps, N_paths);
        auto t2 = std::chrono::high_resolution_clock::now();

        double dt = std::chrono::duration<double>(t2 - t1).count();

        f << th << "," << S0 << "," << K << "," << r << "," << sigma << ","
            << T << "," << N_steps << "," << N_paths << ","
            << price << "," << dt << "\n";

        std::cout << "[OMP] Threads = " << std::setw(2) << th
            << " | Temps = " << dt
            << " | Prix = " << price << "\n";
    }
#endif

    f.close();

    std::cout << "\n[INFO] Résultats écrits dans resultats_lsmc.csv\n";
    return 0;
}
//test commit