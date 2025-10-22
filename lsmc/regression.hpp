/**
 * regression.hpp
 * Régression polynomiale OLS de degré 2
 */
#pragma once
#include <vector>

namespace Regression {
    // Calcule les coefficients d’une régression polynomiale (degré 2)
    std::vector<double> ols_poly2(const std::vector<double>& x, const std::vector<double>& y);
    // Prédit les valeurs selon les coefficients
    std::vector<double> predict_poly2(const std::vector<double>& x, const std::vector<double>& coef);
}
