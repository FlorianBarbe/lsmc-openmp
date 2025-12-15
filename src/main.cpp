// main.cpp — Benchmark automatique LSMC (sans arguments)

#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>
#include <cmath>

#include "lsmc.hpp"
#include "gbm.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

int main() {

    // ==================================================
    // 1) Paramètres modèle FIXES
    // ==================================================
    const double S0 = 100.0;
    const double K = 100.0;
    const double r = 0.05;
    const double sigma = 0.20;
    const double T = 1.0;

#ifdef _OPENMP
    cout << "[INFO] OpenMP actif (" << omp_get_max_threads() << " threads max)\n";
#else
    cout << "[INFO] OpenMP NON actif (séquentiel)\n";
#endif

    cout << fixed << setprecision(6);

    // ==================================================
    // 2) Listes ARBITRAIRES de benchmark
    // ==================================================
    vector<int> steps_list = {
        50, 100, 500, 1000, 5000, 10000
    };

    vector<int> paths_list = {
        100, 500, 1000, 2000, 5000, 10000
    };

    // ==================================================
    // 3) Threads testés
    // ==================================================
    vector<int> thread_list = { 1 };

#ifdef _OPENMP
    if (omp_get_max_threads() >= 2)  thread_list.push_back(2);
    if (omp_get_max_threads() >= 4)  thread_list.push_back(4);
    if (omp_get_max_threads() >= 8)  thread_list.push_back(8);
    if (omp_get_max_threads() >= 16) thread_list.push_back(16);
#endif

    // ==================================================
    // 4) CSV
    // ==================================================
    const string csv_file = "resultats_lsmc.csv";
    bool exists = filesystem::exists(csv_file);

    ofstream f(csv_file, ios::app);

    if (!exists) {
        f << "Threads,S0,K,r,sigma,T,N_steps,N_paths,Prix,Temps,Speedup,RelError,Score\n";
    }

    // ==================================================
    // 5) Benchmark complet
    // ==================================================
    for (int N_steps : steps_list) {
        for (int N_paths : paths_list) {

            cout << "\n===============================================\n";
            cout << "[TEST] N_steps = " << N_steps
                << ", N_paths = " << N_paths << "\n";

            // -------------------------
            // Séquentiel (référence)
            // -------------------------
            double price_seq = 0.0;
            double time_seq = 0.0;

            {
                auto t1 = chrono::high_resolution_clock::now();
                price_seq = LSMC::priceAmericanPut(
                    S0, K, r, sigma, T, N_steps, N_paths
                );
                auto t2 = chrono::high_resolution_clock::now();
                time_seq = chrono::duration<double>(t2 - t1).count();
            }

            cout << "[SEQ] Temps = " << time_seq
                << " | Prix = " << price_seq << "\n";

            f << 1 << "," << S0 << "," << K << "," << r << "," << sigma << ","
                << T << "," << N_steps << "," << N_paths << ","
                << price_seq << "," << time_seq << ","
                << 1.0 << "," << 0.0 << "," << 1.0 << "\n";

#ifdef _OPENMP
            // -------------------------
            // OpenMP
            // -------------------------
            for (int th : thread_list) {

                if (th == 1) continue;

                omp_set_num_threads(th);

                auto t1 = chrono::high_resolution_clock::now();
                double price = LSMC::priceAmericanPut(
                    S0, K, r, sigma, T, N_steps, N_paths
                );
                auto t2 = chrono::high_resolution_clock::now();
                double dt = chrono::duration<double>(t2 - t1).count();

                double speedup = time_seq / dt;
                double err_rel = abs(price - price_seq)
                    / max(abs(price_seq), 1e-12);

                double score = speedup / (1.0 + 10.0 * err_rel);

                cout << "[OMP] " << th << " threads"
                    << " | Temps = " << dt
                    << " | Speedup = " << speedup
                    << " | Prix = " << price
                    << " | Err_rel = " << err_rel
                    << " | Score = " << score << "\n";

                f << th << "," << S0 << "," << K << "," << r << "," << sigma << ","
                    << T << "," << N_steps << "," << N_paths << ","
                    << price << "," << dt << ","
                    << speedup << "," << err_rel << "," << score << "\n";
            }
#endif
        }
    }

    f.close();
    cout << "\n[INFO] Benchmark terminé → resultats_lsmc.csv\n";
    return 0;
}
