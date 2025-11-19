/**
 * Projet LSMC – Module Regression (OLS)
 * -------------------------------------
 * Rôle : effectuer une régression linéaire simple par moindres carrés ordinaires.
 */

#pragma once
#include <vector>

class Regression {
public:
    // Calcule les coefficients d'une régression polynomiale (degré 2)
    static std::vector<double> ols(const std::vector<double>& X, const std::vector<double>& Y);

    // Évalue la régression aux points donnés
    static std::vector<double> predict(const std::vector<double>& X, const std::vector<double>& coeffs);
};
