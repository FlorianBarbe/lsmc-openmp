// main.cpp — LSMC runner: single-run (UI) + benchmark optionnel
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <vector>
#include <iomanip>
#include <cmath>
#include <unordered_map>
#include <atomic>
#include <csignal>

#include "lsmc.hpp"
#include "gbm.hpp"
#include "../runtime_flags.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

// Flags globaux partagés avec lsmc.cpp
std::atomic<bool> g_stop{ false };
bool g_dump_paths = false;

static void on_signal(int) { g_stop.store(true); }

// ------------------------------
// Parsing simple des arguments
// ------------------------------
static unordered_map<string, string> parse_args(int argc, char** argv) {
    unordered_map<string, string> m;
    for (int i = 1; i < argc; ++i) {
        string k = argv[i];
        if (k.rfind("--", 0) == 0) {
            string key = k.substr(2);
            if (i + 1 < argc) {
                string v = argv[i + 1];
                if (v.rfind("--", 0) != 0) {
                    m[key] = v;
                    ++i;
                }
                else {
                    m[key] = "1";
                }
            }
            else {
                m[key] = "1";
            }
        }
    }
    return m;
}

static double getd(const unordered_map<string, string>& a, const string& key, double def) {
    auto it = a.find(key);
    if (it == a.end()) return def;
    try { return stod(it->second); }
    catch (...) { return def; }
}

static int geti(const unordered_map<string, string>& a, const string& key, int def) {
    auto it = a.find(key);
    if (it == a.end()) return def;
    try { return stoi(it->second); }
    catch (...) { return def; }
}

int main(int argc, char** argv) {

    // Handlers STOP (Ctrl+C / Ctrl+Break)
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
#ifdef _WIN32
    std::signal(SIGBREAK, on_signal); // CTRL+BREAK sur Windows
#endif

    auto args = parse_args(argc, argv);

    // Paramètres modèle
    const double S0 = getd(args, "S0", 100.0);
    const double K = getd(args, "K", 100.0);
    const double r = getd(args, "r", 0.05);
    const double sigma = getd(args, "sigma", 0.20);
    const double T = getd(args, "T", 1.0);

    // Paramètres UI
    const int bench = geti(args, "bench", 1);          // --bench 0 => single-run
    const int N_steps = geti(args, "N_steps", 500);      // UI
    const int N_paths = geti(args, "N_paths", 2000);     // UI
    g_dump_paths = (geti(args, "dump_paths", 0) != 0);

#ifdef _OPENMP
    cout << "[INFO] OpenMP actif (" << omp_get_max_threads() << " threads max)\n";
#else
    cout << "[INFO] OpenMP NON actif (séquentiel)\n";
#endif

    cout << fixed << setprecision(6);
    cout.setf(std::ios::unitbuf); // flush live pour l'UI

    cout << "[INFO] Parametres: "
        << "S0=" << S0 << ", K=" << K << ", r=" << r
        << ", sigma=" << sigma << ", T=" << T << "\n";

    cout << "[INFO] Mode: " << (bench == 0 ? "single-run" : "benchmark")
        << " | N_steps=" << N_steps << " | N_paths=" << N_paths
        << " | dump_paths=" << (g_dump_paths ? "1" : "0") << "\n";

    // CSV (reset à chaque run)
    const string csv_file = "resultats_lsmc.csv";
    ofstream f(csv_file, ios::out | ios::trunc);
    if (!f) {
        cerr << "[ERROR] Impossible d'ouvrir " << csv_file << "\n";
        return 1;
    }
    f.setf(std::ios::unitbuf);
    f << "Threads,S0,K,r,sigma,T,N_steps,N_paths,Prix,Temps,Speedup,RelError,Score\n";

    // ==================================================
    // A) SINGLE-RUN (utilisé par l'UI)
    // ==================================================
    if (bench == 0) {

#ifdef _OPENMP
        omp_set_num_threads(1); // référence stable
#endif
        auto t1 = chrono::high_resolution_clock::now();
        double price = LSMC::priceAmericanPut(S0, K, r, sigma, T, N_steps, N_paths);
        auto t2 = chrono::high_resolution_clock::now();
        double dt = chrono::duration<double>(t2 - t1).count();

        cout << "[RUN] Temps = " << dt << " | Prix = " << price << "\n";

        // Speedup/RelError/Score non pertinents en single-run => 1/0/1
        f << 1 << "," << S0 << "," << K << "," << r << "," << sigma << ","
            << T << "," << N_steps << "," << N_paths << ","
            << price << "," << dt << ","
            << 1.0 << "," << 0.0 << "," << 1.0 << "\n";

        if (g_stop.load()) cout << "[INFO] STOP reçu -> sortie propre.\n";
        cout << "[INFO] Terminé → " << csv_file << "\n";
        return 0;
    }

    // ==================================================
    // B) BENCHMARK (si vous lancez sans --bench 0)
    // ==================================================
    vector<int> steps_list = { 50, 100, 500, 1000, 5000, 10000 };
    vector<int> paths_list = { 100, 500, 1000, 2000, 5000, 10000 };

    vector<int> thread_list = { 1 };
#ifdef _OPENMP
    if (omp_get_max_threads() >= 2)  thread_list.push_back(2);
    if (omp_get_max_threads() >= 4)  thread_list.push_back(4);
    if (omp_get_max_threads() >= 8)  thread_list.push_back(8);
    if (omp_get_max_threads() >= 16) thread_list.push_back(16);
#endif

    // Ici on respecte la limite demandée via --N_steps/--N_paths
    for (int Ns : steps_list) {
        if (g_stop.load()) break;
        if (Ns > N_steps) break; // ne dépasse jamais l'UI

        for (int Np : paths_list) {
            if (g_stop.load()) break;
            if (Np > N_paths) break; // ne dépasse jamais l'UI

            cout << "\n===============================================\n";
            cout << "[TEST] N_steps = " << Ns << ", N_paths = " << Np << "\n";

            double price_seq = 0.0;
            double time_seq = 0.0;

#ifdef _OPENMP
            omp_set_num_threads(1);
#endif
            auto t1 = chrono::high_resolution_clock::now();
            price_seq = LSMC::priceAmericanPut(S0, K, r, sigma, T, Ns, Np);
            auto t2 = chrono::high_resolution_clock::now();
            time_seq = chrono::duration<double>(t2 - t1).count();

            cout << "[SEQ] Temps = " << time_seq << " | Prix = " << price_seq << "\n";
            f << 1 << "," << S0 << "," << K << "," << r << "," << sigma << ","
                << T << "," << Ns << "," << Np << ","
                << price_seq << "," << time_seq << ","
                << 1.0 << "," << 0.0 << "," << 1.0 << "\n";

#ifdef _OPENMP
            for (int th : thread_list) {
                if (g_stop.load()) break;
                if (th == 1) continue;

                omp_set_num_threads(th);

                auto u1 = chrono::high_resolution_clock::now();
                double price = LSMC::priceAmericanPut(S0, K, r, sigma, T, Ns, Np);
                auto u2 = chrono::high_resolution_clock::now();
                double dt = chrono::duration<double>(u2 - u1).count();

                double speedup = (dt > 0.0) ? (time_seq / dt) : 0.0;
                double err_rel = abs(price - price_seq) / max(abs(price_seq), 1e-12);
                double score = speedup / (1.0 + 10.0 * err_rel);

                cout << "[OMP] " << th << " threads"
                    << " | Temps = " << dt
                    << " | Speedup = " << speedup
                    << " | Prix = " << price
                    << " | Err_rel = " << err_rel
                    << " | Score = " << score << "\n";

                f << th << "," << S0 << "," << K << "," << r << "," << sigma << ","
                    << T << "," << Ns << "," << Np << ","
                    << price << "," << dt << ","
                    << speedup << "," << err_rel << "," << score << "\n";
            }
#endif
        }
    }

    if (g_stop.load()) cout << "[INFO] STOP reçu -> sortie propre.\n";
    cout << "\n[INFO] Benchmark terminé → " << csv_file << "\n";
    return 0;
}
