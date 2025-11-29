//main.cpp — version pour séquentiel + multi-threads

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

	//si pas assez d'arguments
    if (argc < 8) {
        std::cerr << "Usage : lsmc.exe S0 K r sigma T N_steps N_paths\n";
        return 1;
    }
	//pour verif le nombre d'arguments
    std::cout << "argc = " << argc << "\n";

	// conversion des arguments en variables
    double S0 = std::stod(argv[1]);
    double K = std::stod(argv[2]);
    double r = std::stod(argv[3]);
    double sigma = std::stod(argv[4]);
    double T = std::stod(argv[5]);
    int N_steps = std::stoi(argv[6]);
    int N_paths = 200000;//std::stoi(argv[7]);

	// Affichage des paramètres
    std::cout << std::fixed << std::setprecision(6);

#ifdef _OPENMP
    std::cout << "[INFO] OpenMP actif ("
        << omp_get_max_threads()
        << " threads disponibles)\n";
#else
    std::cout << "[INFO] OpenMP non actif — mode sequentiel\n";
#endif

    std::cout << "[INFO] Simulation de "
        << N_paths << " trajectoires, "
        << N_steps << " pas temporels\n";

	//simulation des trajectoires GBM et export CSV
    auto paths = GBM::simulatePaths(S0, r, sigma, T, N_steps, N_paths);
    GBM::exportCSV(paths, "trajectoires_gbm.csv");
    std::string csv_file = "resultats_lsmc.csv";
    bool file_exists = std::filesystem::exists(csv_file);
    std::ofstream f(csv_file, std::ios::app);
	// écriture de l'en-tête si le fichier n'existe pas
    if (!file_exists) {
        f << "Threads,S0,K,r,sigma,T,N_steps,N_paths,Prix,Temps\n";
    }

	// execution séquentielle pour référence
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


	// tests multi-threads avec OpenMP
#ifdef _OPENMP
    std::vector<int> thread_list = { 2, 4, 8, 16 };

    for (int th : thread_list) {
        omp_set_num_threads(th);

        //start
        auto t1 = std::chrono::high_resolution_clock::now();
        //calcul
        double price = LSMC::priceAmericanPut(S0, K, r, sigma, T, N_steps, N_paths);
        //fin
        auto t2 = std::chrono::high_resolution_clock::now();
		//calcul du temps de calcul
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

    std::cout << "\n[INFO] Resultats ecrits dans resultats_lsmc.csv\n";
    return 0;
}
//test commit