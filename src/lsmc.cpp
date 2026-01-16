/**
 * Implémentation du module Longstaff–Schwartz (parallélisée avec OpenMP)
 */

#include "lsmc.hpp"
#include "runtime_flags.hpp"
#include <fstream>

using namespace std;

static inline void solve3x3(double m[3][3], double b[3], double x[3]) {
  for (int k = 0; k < 3; ++k) {
    // Pivot principal
    double piv = m[k][k];

    // Si pivot trop petit -> risque d'instabilité -> petite régularisation
    if (fabs(piv) < 1e-14)
      piv = (piv >= 0.0 ? 1e-14 : -1e-14);

    double inv_piv = 1.0 / piv;

    // Normalisation de la ligne k
    for (int j = k; j < 3; ++j)
      m[k][j] *= inv_piv;
    b[k] *= inv_piv;

    // Élimination dans les autres lignes
    for (int i = 0; i < 3; ++i) {
      if (i == k)
        continue;

      double factor = m[i][k];
      for (int j = k; j < 3; ++j)
        m[i][j] -= factor * m[k][j];
      b[i] -= factor * b[k];
    }
  }

  // Récupération de x
  for (int i = 0; i < 3; ++i)
    x[i] = b[i];
}

// pour accéder aux indices beaucoup plus facilement
inline int idx(int i, int t, int N_steps) { return i * (N_steps + 1) + t; }

double LSMC::priceAmericanPut(double S0, double K, double r, double sigma,
                              double T, int N_steps, int N_paths, int seed) {
  double dt = T / N_steps;
  double discount = exp(-r * dt);

  vector<double> paths(N_paths * (N_steps + 1));
  vector<double> payoff(N_paths * (N_steps + 1));

  GBM gbm(S0, r, sigma, T, N_steps);

  gbm.simulatePaths(paths.data(), S0, r, sigma, T, N_steps, N_paths, seed);
  if (g_dump_paths) {
    ofstream outfile("paths.csv");
    if (outfile.is_open()) {
      outfile << "path_id";
      for (int t = 0; t <= N_steps; ++t)
        outfile << ",t" << t;
      outfile << "\n";

      for (int i = 0; i < N_paths; ++i) {
        outfile << i;
        for (int t = 0; t <= N_steps; ++t) {
          outfile << "," << paths[idx(i, t, N_steps)];
        }
        outfile << "\n";
      }
    }
  }

#pragma omp parallel for schedule(static)
  for (int i = 0; i < N_paths; ++i)
    for (int t = 0; t <= N_steps; ++t) {
      const double S = paths[idx(i, t, N_steps)];
      payoff[idx(i, t, N_steps)] = max(K - S, 0.0);
    }

  // 3. Backward induction parallélisée

  // cashflow à maturité=payoff final
  vector<double> cashflows(N_paths);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < N_paths; ++i)
    cashflows[i] = payoff[idx(i, N_steps, N_steps)];

  // Boucle principale backward

  for (int t = N_steps - 1; t > 0; --t) {
    double a00 = 0, a01 = 0, a02 = 0;
    double a11 = 0, a12 = 0, a22 = 0;
    double b0 = 0, b1 = 0, b2 = 0;

#pragma omp parallel for reduction(+ : a00, a01, a02, a11, a12, a22, b0, b1,   \
                                       b2) schedule(static)
    for (int i = 0; i < N_paths; i++) {
      // ne considérer que les paths "dans la monnaie"
      if (payoff[idx(i, t, N_steps)] > 0.0) {
        const double S = paths[idx(i, t, N_steps)];
        const double Y = cashflows[i] * discount;

        const double phi0 = 1.0;
        const double phi1 = S;
        const double phi2 = S * S;

        // AtA
        a00 += phi0 * phi0;
        a01 += phi0 * phi1;
        a02 += phi0 * phi2;
        a11 += phi1 * phi1;
        a12 += phi1 * phi2;
        a22 += phi2 * phi2;

        // Aty
        b0 += phi0 * Y;
        b1 += phi1 * Y;
        b2 += phi2 * Y;
      }
    }

    // Si a00=0, aucune trajectoire ITM : on continue simplement
    if (a00 == 0.0) {
#pragma omp parallel for schedule(static)
      for (int i = 0; i < N_paths; ++i)
        cashflows[i] *= discount;
      continue;
    }

    double M[3][3]{{a00, a01, a02}, {a01, a11, a12}, {a02, a12, a22}};

    double B[3] = {b0, b1, b2};
    double beta[3];

    // résolution rapide du petit système
    solve3x3(M, B, beta);

    const double beta0 = beta[0];
    const double beta1 = beta[1];
    const double beta2 = beta[2];

    // MaJ des payoffs parallélisée

#pragma omp parallel for schedule(static)
    for (int i = 0; i < N_paths; ++i) {
      const double immediate = payoff[idx(i, t, N_steps)];
      if (immediate > 0.0) {
        const double S = paths[idx(i, t, N_steps)];
        const double continuation = beta0 + beta1 * S + beta2 * S * S;
        if (immediate > continuation)
          cashflows[i] = immediate;
        else
          cashflows[i] *= discount;
      } else {
        // Hors de la monnaie : impossible d'exercer
        cashflows[i] *= discount;
      }
    }
  }

  // 4. Moyenne finale

  double mean = 0.0;
#pragma omp parallel for reduction(+ : mean)
  for (int i = 0; i < N_paths; ++i)
    mean += cashflows[i];

  return mean / N_paths;
}
