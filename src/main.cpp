// main.cpp — Benchmark fixe : N_steps = {100, 1000, 10000}, N_paths = {100, 1000, 10000}

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

int main(int argc, char** argv) {

    if (argc < 6) {
        cerr << "Usage : lsmc.exe S0 K r sigma T\n";
        return 1;
    }

    // Paramètres modèle (uniquement)
    double S0 = stod(argv[1]);
    double K = stod(argv[2]);
    double r = stod(argv[3]);
    double sigma = stod(argv[4]);
    double T = stod(argv[5]);

#ifdef _OPENMP
    cout << "[INFO] OpenMP actif (" << omp_get_max_threads() << " threads dispo)\n";
#else
    cout << "[INFO] OpenMP NON actif — mode séquentiel seulement\n";
#endif

    cout << fixed << setprecision(6);

    // --------------------------------------------------
    // 1) Listes FIXES demandées
    // --------------------------------------------------
    vector<int> steps_list = { 100, 1000, 10000 };
    vector<int> paths_list = { 100, 1000, 10000 };

    // --------------------------------------------------
    // 2) Threads testés
    // --------------------------------------------------
    vector<int> thread_list = { 1 };

#ifdef _OPENMP
    if (omp_get_max_threads() >= 2)  thread_list.push_back(2);
    if (omp_get_max_threads() >= 4)  thread_list.push_back(4);
    if (omp_get_max_threads() >= 8)  thread_list.push_back(8);
    if (omp_get_max_threads() >= 16) thread_list.push_back(16);
#endif

    // --------------------------------------------------
    // 3) Ouverture CSV
    // --------------------------------------------------
    const string csv_file = "resultats_lsmc.csv";
    bool exists = filesystem::exists(csv_file);

    ofstream f(csv_file, ios::app);

    if (!exists) {
        f << "Threads,S0,K,r,sigma,T,N_steps,N_paths,Prix,Temps,Speedup,RelError,Score\n";
    }

    // --------------------------------------------------
    // 4) Benchmark complet
    // --------------------------------------------------
    for (int N_steps : steps_list) {
        for (int N_paths : paths_list) {

            cout << "\n===============================================\n";
            cout << "[CONFIG TEST] N_steps = " << N_steps
                << ", N_paths = " << N_paths << "\n";

            // =========================
            // a) Séquentiel (référence)
            // =========================
            double price_seq = 0.0;
            double time_seq = 0.0;

            {
                auto t1 = chrono::high_resolution_clock::now();
                price_seq = LSMC::priceAmericanPut(S0, K, r, sigma, T,
                    N_steps, N_paths);
                auto t2 = chrono::high_resolution_clock::now();
                time_seq = chrono::duration<double>(t2 - t1).count();
            }

            f << 1 << "," << S0 << "," << K << "," << r << "," << sigma << ","
                << T << "," << N_steps << "," << N_paths << ","
                << price_seq << "," << time_seq << ","
                << 1.0 << "," << 0.0 << "," << 1.0 << "\n";

            cout << "[SEQ] Temps = " << time_seq
                << " | Prix = " << price_seq << "\n";

#ifdef _OPENMP
            // =========================
            // b) Parallélisme OpenMP
            // =========================
            for (int th : thread_list) {

                if (th == 1) continue;

                omp_set_num_threads(th);

                auto t1 = chrono::high_resolution_clock::now();
                double price = LSMC::priceAmericanPut(S0, K, r, sigma, T,
                    N_steps, N_paths);
                auto t2 = chrono::high_resolution_clock::now();
                double dt = chrono::duration<double>(t2 - t1).count();

                double speedup = time_seq / dt;

                double denom = fabs(price_seq);// max(1e-12, fabs(price_seq));
                double err_abs = std::abs(price - price_seq);
                double err_rel = err_abs / std::max(std::abs(price_seq), 1e-12);


            
                double score = speedup / (1.0 + 10* err_rel);

                cout << "[OMP] " << th << " threads"
                    << " | Temps = " << dt
                    << " | Speedup = " << speedup
                    << " | Prix = " << price
                    << setprecision(6)
                    << "Err_abs = " << err_abs
                    << " | Err_rel = " << err_rel << "\n"
                    << " | Score = " << score
                    << "\n";

                f << th << "," << S0 << "," << K << "," << r << "," << sigma << ","
                    << T << "," << N_steps << "," << N_paths << ","
                    << price << "," << dt << ","
                    << speedup << "," << err_rel << "," << score << "\n";
            }
#endif
        }
    }

    f.close();
    cout << "\n[INFO] Resultats écrits dans resultats_lsmc.csv\n";
    return 0;
}
