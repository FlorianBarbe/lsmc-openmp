#pragma once
#include <vector>
#include <string>
#include "rng.hpp"
using namespace std;
// -----------------------------------------------------------------------------
// Classe GBM (Geometric Brownian Motion) qui regroupe les fonctions liées au mouvement brownien géométrique.
// Elle sert à :
//   1) simuler un ensemble de trajectoires du sous-jacent selon le modèle GBM,
//   2) stocker ces trajectoires dans une matrice paths[ N_paths ][ N_steps+1 ],
//   3) exporter proprement ces trajectoires dans un fichier CSV pour Streamlit.
//
// Elle fournit donc les données brutes nécessaires au LSMC.
// -----------------------------------------------------------------------------


class GBM {
private:
    double S0;
    double r;
    double sigma;
    double T;
    int N_steps;
public:
    // Constructeur : paramètres du modèle
    GBM(double S0, double r, double sigma, double T, int N_steps);

    // Simule UNE trajectoire
    void simulate_path(RNG& rng, double* path_out);

    // Simule PLUSIEURS trajectoires
    static void simulatePaths(double* paths,double S0, double r, double sigma, double T, int N_steps, int N_paths);

    // Exporte un tableau de trajectoires dans un CSV
    static void exportCSV(
        const std::vector<std::vector<double>>& paths,
        const std::string& filename
    );
};
