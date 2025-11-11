#include <iostream>
#include <chrono>
#include "lsmc.hpp"

int main() {
    double S0 = 100.0;
    double K = 100.0;
    double r = 0.05;
    double sigma = 0.2;
    double T = 1.0;
    int N_steps = 50;
    int N_paths = 20000; // plus de trajectoires pour que la différence soit visible

    // ========================
    // 1. Exécution SANS OpenMP
    // ========================
    std::cout << "=== Version séquentielle ===" << std::endl;
    auto start_seq = std::chrono::high_resolution_clock::now();

    double price_seq = LSMC::priceAmericanPut(S0, K, r, sigma, T, N_steps, N_paths);

    auto end_seq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_seq = end_seq - start_seq;

    std::cout << "Prix estimé : " << price_seq << std::endl;
    std::cout << "Temps d'exécution (sans OpenMP) : " << time_seq.count() << " s" << std::endl << std::endl;

    // ========================
    // 2. Exécution AVEC OpenMP
    // ========================
    std::cout << "=== Version parallèle (OpenMP) ===" << std::endl;

    // Active OpenMP
#ifdef _OPENMP
    std::cout << "OpenMP détecté, exécution multi-thread." << std::endl;
#else
    std::cout << "OpenMP non activé, exécution simple thread." << std::endl;
#endif

    auto start_par = std::chrono::high_resolution_clock::now();

    double price_par = LSMC::priceAmericanPut(S0, K, r, sigma, T, N_steps, N_paths);

    auto end_par = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_par = end_par - start_par;

    std::cout << "Prix estimé : " << price_par << std::endl;
    std::cout << "Temps d'exécution (avec OpenMP) : " << time_par.count() << " s" << std::endl;

    return 0;
}
