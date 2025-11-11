/**
 * Implémentation du module Regression (OLS)
 */

#include "regression.hpp"
#include <vector>
#include <cmath>

std::vector<double> Regression::ols(const std::vector<double>& X, const std::vector<double>& Y) {
    // Calcul des sommes nécessaires
    double n = static_cast<double>(X.size());
    double sumX = 0, sumX2 = 0, sumX3 = 0, sumX4 = 0;
    double sumY = 0, sumXY = 0, sumX2Y = 0;

    for (size_t i = 0; i < X.size(); ++i) {
        double x = X[i];
        double y = Y[i];
        sumX += x;
        sumX2 += x * x;
        sumX3 += x * x * x;
        sumX4 += x * x * x * x;
        sumY += y;
        sumXY += x * y;
        sumX2Y += x * x * y;
    }

    // Résolution du système linéaire 3x3 pour a0, a1, a2
    double D = n * (sumX2 * sumX4 - sumX3 * sumX3)
        - sumX * (sumX * sumX4 - sumX2 * sumX3)
        + sumX2 * (sumX * sumX3 - sumX2 * sumX2);

    double Da0 = sumY * (sumX2 * sumX4 - sumX3 * sumX3)
        - sumX * (sumXY * sumX4 - sumX3 * sumX2Y)
        + sumX2 * (sumXY * sumX3 - sumX2 * sumX2Y);

    double Da1 = n * (sumXY * sumX4 - sumX3 * sumX2Y)
        - sumY * (sumX * sumX4 - sumX2 * sumX3)
        + sumX2 * (sumX * sumX2Y - sumX2 * sumXY);

    double Da2 = n * (sumX2 * sumX2Y - sumX3 * sumXY)
        - sumX * (sumX * sumX2Y - sumX2 * sumXY)
        + sumY * (sumX * sumX3 - sumX2 * sumX2);

    return { Da0 / D, Da1 / D, Da2 / D };
}

std::vector<double> Regression::predict(const std::vector<double>& X, const std::vector<double>& coeffs) {
    std::vector<double> Y_pred;
    Y_pred.reserve(X.size());
    for (double x : X)
        Y_pred.push_back(coeffs[0] + coeffs[1] * x + coeffs[2] * x * x);
    return Y_pred;
}
