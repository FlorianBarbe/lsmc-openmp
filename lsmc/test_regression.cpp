/**
 * Test unitaire : validation du module Regression
 * ------------------------------------------------
 * Objectif : vérifier que la régression polynomiale d'ordre 2
 * retrouve les bons coefficients pour un jeu de données connu.
 */

#include <iostream>
#include <vector>
#include "regression.hpp"

int main() {
    // On crée des données simulées suivant y = 2 + 3x + 4x²
    std::vector<double> x = { -2, -1, 0, 1, 2 };
    std::vector<double> y;
    for (double xi : x)
        y.push_back(2 + 3 * xi + 4 * xi * xi);

    // On calcule les coefficients via notre fonction OLS
    auto coeffs = Regression::ols_poly2(x, y);

    std::cout << "=== Test du module Regression ===" << std::endl;
    std::cout << "Coefficients estimés : ";
    for (double c : coeffs) std::cout << c << " ";
    std::cout << std::endl;

    // Valeurs théoriques
    double a0 = 2, a1 = 3, a2 = 4;

    double err_a0 = std::abs(coeffs[0] - a0);
    double err_a1 = std::abs(coeffs[1] - a1);
    double err_a2 = std::abs(coeffs[2] - a2);

    std::cout << "Erreur absolue a0 : " << err_a0 << std::endl;
    std::cout << "Erreur absolue a1 : " << err_a1 << std::endl;
    std::cout << "Erreur absolue a2 : " << err_a2 << std::endl;

    if (err_a0 < 1e-10 && err_a1 < 1e-10 && err_a2 < 1e-10)
        std::cout << "[OK] Test réussi (régression correcte)" << std::endl;
    else
        std::cout << "[ERREUR] Test échoué (mauvais coefficients)" << std::endl;

    return 0;
}
