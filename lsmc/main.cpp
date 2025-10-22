/**
 * main.cpp
 * ------------------------------------------------------------
 * Programme principal du projet LSMC (Longstaff–Schwartz Monte Carlo)
 * ------------------------------------------------------------
 * Objectif :
 *   Estimer le prix d’un put américain en utilisant la méthode
 *   de Longstaff–Schwartz (régression polynomiale + Monte Carlo).
 *
 * Fonctionnement :
 *   - Génère un grand nombre de trajectoires du sous-jacent (GBM)
 *   - Applique une régression OLS pour la valeur de continuation
 *   - Compare payoff immédiat / continuation pour déterminer l’exercice
 *
 * Parallélisation :
 *   La simulation Monte Carlo est parallélisée avec OpenMP.
 *   Le nombre de threads utilisés dépend de la variable d’environnement :
 *     OMP_NUM_THREADS = k
 *   (définie automatiquement par le script Python ou manuellement)
 *
 * Utilisation :
 *   ./lsmc.exe [N_paths]
 *   (si aucun argument n’est fourni, N_paths = 20000 par défaut)
 */

#include <iostream>
#include <chrono>
#include <string>
#include "lsmc.hpp"
#include "gbm.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char** argv) {
    // --- Paramètres du modèle ---
    double S0 = 100.0;     // Prix initial
    double K = 100.0;      // Strike
    double r = 0.05;       // Taux d'intérêt
    double sigma = 0.2;    // Volatilité
    double T = 1.0;        // Maturité (en années)
    int N_steps = 50;      // Nombre de pas temporels
    int N_paths = 20000;   // Nombre de trajectoires simulées

    // Si un argument est fourni, il remplace N_paths
    if (argc > 1)
        N_paths = std::stoi(argv[1]);

    // --- Informations sur le parallélisme ---
#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    std::cout << "[INFO] OpenMP activé (" << nthreads << " threads disponibles)" << std::endl;
#else
    std::cout << "[INFO] OpenMP non activé (exécution séquentielle)" << std::endl;
#endif

    std::cout << "[INFO] Simulation de " << N_paths << " trajectoires avec "
        << N_steps << " pas de temps..." << std::endl;

    // === Simulation GBM ===
    std::cout << "[INFO] Simulation de " << N_paths << " trajectoires..." << std::endl;
    auto paths = GBM::simulatePaths(S0, r, sigma, T, N_steps, N_paths);

    // Export CSV pour visualisation Python
    GBM::exportCSV(paths, "../../output/trajectoires_gbm.csv");


    // --- Calcul du prix ---
    auto start = std::chrono::high_resolution_clock::now();

    double price = LSMC::priceAmericanPut(S0, K, r, sigma, T, N_steps, N_paths);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // --- Affichage des résultats ---
    std::cout << "\n=== Résultats ===" << std::endl;
    std::cout << "Prix estimé du put américain : " << price << std::endl;
    std::cout << "Temps d'exécution : " << elapsed.count() << " secondes" << std::endl;

#ifdef _OPENMP
    std::cout << "[INFO] Calcul effectué en parallèle sur " << nthreads << " threads." << std::endl;
#else
    std::cout << "[INFO] Calcul effectué en mode séquentiel." << std::endl;
#endif

    std::cout << "=================\n" << std::endl;

    return 0;
}


