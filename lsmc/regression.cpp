#include "regression.hpp"
#include <vector>
#include <numeric>

std::vector<double> Regression::ols_poly2(const std::vector<double>& x, const std::vector<double>& y) {
    int n = x.size();
    double Sx = 0, Sx2 = 0, Sx3 = 0, Sx4 = 0;
    double Sy = 0, Sxy = 0, Sx2y = 0;

    for (int i = 0; i < n; ++i) {
        double xi = x[i], yi = y[i];
        double xi2 = xi * xi;
        Sx += xi;
        Sx2 += xi2;
        Sx3 += xi2 * xi;
        Sx4 += xi2 * xi2;
        Sy += yi;
        Sxy += xi * yi;
        Sx2y += xi2 * yi;
    }

    double denom = n * (Sx2 * Sx4 - Sx3 * Sx3)
        - Sx * (Sx * Sx4 - Sx2 * Sx3)
        + Sx2 * (Sx * Sx3 - Sx2 * Sx2);

    double a0 = (Sy * (Sx2 * Sx4 - Sx3 * Sx3)
        - Sx * (Sxy * Sx4 - Sx3 * Sx2y)
        + Sx2 * (Sxy * Sx3 - Sx2 * Sx2y)) / denom;

    double a1 = (n * (Sxy * Sx4 - Sx3 * Sx2y)
        - Sy * (Sx * Sx4 - Sx2 * Sx3)
        + Sx2 * (Sx * Sx2y - Sx2 * Sxy)) / denom;

    double a2 = (n * (Sx2 * Sx2y - Sx3 * Sxy)
        - Sx * (Sx * Sx2y - Sx2 * Sxy)
        + Sy * (Sx * Sx3 - Sx2 * Sx2)) / denom;

    return { a0, a1, a2 };
}

std::vector<double> Regression::predict_poly2(const std::vector<double>& x, const std::vector<double>& coef) {
    std::vector<double> y(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        y[i] = coef[0] + coef[1] * x[i] + coef[2] * x[i] * x[i];
    return y;
}
#pragma once
