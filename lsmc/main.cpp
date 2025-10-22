/**
 * main.cpp — version intégrée Streamlit (avec tests OpenMP)
 * ------------------------------------------------------------
 * Exécutable principal appelé depuis l’interface Streamlit.
 * Lit les paramètres passés en argument, exécute la simulation
 * LSMC pour un put américain, mesure les temps d’exécution
 * séquentiel et parallèle, et exporte les résultats au format CSV.
 *
 * Usage :
 *   lsmc.exe S0 K r sigma T N_steps N_paths
 */

#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <vector>
#include <iomanip>
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

    // === Lecture des arguments ===
    double S0 = std::stod(argv[1]);
    double K = std::stod(argv[2]);
    double r = std::stod(argv[3]);
    double sigma = std::stod(argv[4]);
    double T = std::stod(argv[5]);
    int N_steps = std::stoi(argv[6]);
    int N_paths = std::stoi(argv[7]);

    std::cout << std::fixed << std::setprecision(6);
#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    std::cout << "[INFO] OpenMP activé (" << nthreads << " threads disponibles)" << std::endl;
#else
    std::cout << "[INFO] OpenMP non activé — exécution séquentielle uniquement" << std::endl;
#endif

    std::cout << "[INFO] Simulation de " << N_paths << " trajectoires avec "
        << N_steps << " pas de temps...\n";

    // === Simulation GBM ===
    auto paths = GBM::simulatePaths(S0, r, sigma, T, N_steps, N_paths);
    GBM::exportCSV(paths, "trajectoires_gbm.csv");

    // === Mesure séquentielle ===
    auto t1 = std::chrono::high_resolution_clock::now();
    double price_seq = LSMC::priceAmericanPut(S0, K, r, sigma, T, N_steps, N_paths);
    auto t2 = std::chrono::high_resolution_clock::now();
    double time_seq = std::chrono::duration<double>(t2 - t1).count();

    // === Mesure OpenMP (parallèle) ===
#ifdef _OPENMP
    auto t3 = std::chrono::high_resolution_clock::now();
    double price_omp = LSMC::priceAmericanPut(S0, K, r, sigma, T, N_steps, N_paths);
    auto t4 = std::chrono::high_resolution_clock::now();
    double time_omp = std::chrono::duration<double>(t4 - t3).count();
    double speedup = time_seq / time_omp;
#else
    double price_omp = price_seq;
    double time_omp = time_seq;
    double speedup = 1.0;
#endif

    // === Test comparatif multi-threads (1, 4, 16) ===
#ifdef _OPENMP
    std::cout << "\n=== Tests de performance ===\n";
    std::vector<int> thread_tests = { 1, 4, 16 };
    for (int nt : thread_tests) {
        omp_set_num_threads(nt);
        auto start = std::chrono::high_resolution_clock::now();
        double p = LSMC::priceAmericanPut(S0, K, r, sigma, T, N_steps, N_paths);
        auto end = std::chrono::high_resolution_clock::now();
        double t = std::chrono::duration<double>(end - start).count();
        std::cout << "[TEST] Threads = " << std::setw(2) << nt
            << " | Temps = " << std::setw(8) << t << " s"
            << " | Prix = " << std::setw(8) << p << std::endl;
    }
#endif

    // === Résumé console ===
    std::cout << "\n=== Résultats ===\n";
    std::cout << "Prix séquentiel : " << price_seq << " (" << time_seq << " s)\n";
    std::cout << "Prix OpenMP     : " << price_omp << " (" << time_omp << " s)\n";
    std::cout << "Speedup         : " << speedup << "x\n";

    // === Export CSV ===
    std::ofstream f("resultats_lsmc.csv", std::ios::app);
    if (f.tellp() == 0) {
        f << "S0,K,r,sigma,T,N_steps,N_paths,Prix_Seq,Temps_Seq,Prix_OpenMP,Temps_OpenMP,Speedup\n";
    }
    f << S0 << "," << K << "," << r << "," << sigma << "," << T << ","
        << N_steps << "," << N_paths << ","
        << price_seq << "," << time_seq << ","
        << price_omp << "," << time_omp << "," << speedup << "\n";
    f.close();

#ifdef _OPENMP
    std::cout << "[INFO] Calcul effectué en parallèle sur " << nthreads << " threads." << std::endl;
#else
    std::cout << "[INFO] Calcul séquentiel terminé." << std::endl;
#endif

    std::cout << "=============================\n";
    return 0;
}
