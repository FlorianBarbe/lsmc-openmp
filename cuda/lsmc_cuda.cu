// lsmc_cuda.cu
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

// ===========================================
// Helpers CUDA
// ===========================================
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            std::cerr << "CUDA error " << cudaGetErrorString(err)         \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";   \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

// ===========================================
// Kernel 1 : simulation GBM sur GPU
// paths[t * N_paths + i] = prix au temps t pour le path i
// ===========================================
__global__ void simulatePathsKernel(
    double* paths,
    double S0, double r, double sigma,
    double T, int N_steps, int N_paths,
    unsigned long long seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_paths) return;

    double dt = T / (double)(N_steps - 1);
    double drift = (r - 0.5 * sigma * sigma) * dt;
    double vol   = sigma * sqrt(dt);

    // RNG très simplifié (Xorshift) + Box-Muller pour la démo
    unsigned long long state = seed ^ (unsigned long long)i;

    auto nextUniform = [&]() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return (state >> 11) * (1.0 / (9007199254740992.0)); // ~ [0,1)
    };

    auto nextNormal = [&]() {
        double u1 = nextUniform();
        double u2 = nextUniform();
        double r  = sqrt(-2.0 * log(u1 + 1e-12));
        double th = 2.0 * M_PI * u2;
        return r * cos(th);
    };

    // t = 0
    paths[0 * N_paths + i] = S0;
    double S = S0;

    for (int t = 1; t < N_steps; ++t) {
        double Z = nextNormal();
        S = S * exp(drift + vol * Z);
        paths[t * N_paths + i] = S;
    }
}

// ===========================================
// Kernel 2 : payoff à maturité (American put)
// cashflows[i] = payoff(T)
// ===========================================
__global__ void initCashflowsKernel(
    const double* paths,
    double* cashflows,
    double K,
    int N_steps,
    int N_paths)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_paths) return;

    double ST = paths[(N_steps - 1) * N_paths + i];
    double payoff = fmax(K - ST, 0.0);
    cashflows[i] = payoff;
}

// ===========================================
// Kernel 3 : construire les équations normales A^T A et A^T y
// pour la régression LSMC à la date t
//
// Base : [1, S, S^2] => système 3x3
//
// On accumule dans 9 doubles (mat) + 3 doubles (vec) avec atomicAdd.
// ===========================================
__global__ void buildNormalEquationsKernel(
    const double* paths,
    const double* cashflows_next, // valeurs de continuation déjà actualisées
    double K,
    double discount,
    int t,
    int N_paths,
    double* mat_out,   // taille 9
    double* vec_out)   // taille 3
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_paths) return;

    double S = paths[t * N_paths + i];
    double payoff = fmax(K - S, 0.0);

    // On ne considère que les paths in-the-money ET qui ont un payoff futur > 0
    if (payoff <= 0.0) return;

    double Y = cashflows_next[i] * discount; // continuation actualisée

    double phi0 = 1.0;
    double phi1 = S;
    double phi2 = S * S;

    double phi[3] = {phi0, phi1, phi2};

    // A^T A = somme phi_i * phi_j
    // A^T y = somme phi_i * Y
    // mat_out[row * 3 + col]
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            double val = phi[row] * phi[col];
            atomicAdd(&mat_out[row * 3 + col], val);
        }
        double vy = phi[row] * Y;
        atomicAdd(&vec_out[row], vy);
    }
}

// ===========================================
// Device : résolution d'un système 3x3 (Gaussian elimination simple)
// mat : 3x3, vec : 3
// res : 3
// ===========================================
__device__ void solve3x3(double* mat, double* vec, double* res)
{
    // On copie dans une petite matrice locale
    double a[3][3];
    double b[3];
    for (int i = 0; i < 3; ++i) {
        b[i] = vec[i];
        for (int j = 0; j < 3; ++j) {
            a[i][j] = mat[i * 3 + j];
        }
    }

    // Elimination
    for (int k = 0; k < 3; ++k) {
        double piv = a[k][k];
        if (fabs(piv) < 1e-14) {
            // fallback : régularisation triviale
            piv = (piv >= 0 ? 1e-14 : -1e-14);
        }
        double inv_piv = 1.0 / piv;
        for (int j = k; j < 3; ++j) a[k][j] *= inv_piv;
        b[k] *= inv_piv;

        for (int i = 0; i < 3; ++i) {
            if (i == k) continue;
            double factor = a[i][k];
            for (int j = k; j < 3; ++j) {
                a[i][j] -= factor * a[k][j];
            }
            b[i] -= factor * b[k];
        }
    }

    for (int i = 0; i < 3; ++i) res[i] = b[i];
}

// ===========================================
// Kernel 4 : à partir de (A^T A, A^T y), on calcule les coefficients beta[]
// et on décide exercice / continuation
//
// cashflows_t[i] = max( payoff(t), continuation(t) )
// ===========================================
__global__ void updateCashflowsKernel(
    const double* paths,
    const double* cashflows_next,
    double* cashflows_t,
    double K,
    double discount,
    int t,
    int N_paths,
    double* mat, // 9
    double* vec) // 3
{
    // On utilise 1 seul thread pour résoudre 3x3,
    // puis on broadcast les coefficients par __shared__.
    __shared__ double beta_shared[3];

    if (threadIdx.x == 0) {
        double beta[3];
        solve3x3(mat, vec, beta);
        beta_shared[0] = beta[0];
        beta_shared[1] = beta[1];
        beta_shared[2] = beta[2];
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_paths) return;

    double S = paths[t * N_paths + i];
    double payoff = fmax(K - S, 0.0);

    double cont = 0.0;
    if (payoff > 0.0) {
        double phi0 = 1.0;
        double phi1 = S;
        double phi2 = S * S;
        cont = beta_shared[0] * phi0
             + beta_shared[1] * phi1
             + beta_shared[2] * phi2;
    }

    // On compare payoff immédiat vs valeur de continuation
    double future = cashflows_next[i] * discount;
    double value = future;
    if (payoff > cont && payoff > future) {
        value = payoff;
    }

    cashflows_t[i] = value;
}

// ===========================================
// Fonction host : prix American Put en full CUDA
// ===========================================
namespace LSMC_GPU {

double priceAmericanPut(
    double S0, double K, double r, double sigma,
    double T, int N_steps, int N_paths)
{
    // 1) Allocation device
    size_t size_paths = (size_t)N_steps * (size_t)N_paths * sizeof(double);
    size_t size_vec   = (size_t)N_paths * sizeof(double);

    double* d_paths = nullptr;
    double* d_cf_curr = nullptr;
    double* d_cf_next = nullptr;
    double* d_mat = nullptr;
    double* d_vec = nullptr;

    CUDA_CHECK(cudaMalloc(&d_paths,   size_paths));
    CUDA_CHECK(cudaMalloc(&d_cf_curr, size_vec));
    CUDA_CHECK(cudaMalloc(&d_cf_next, size_vec));
    CUDA_CHECK(cudaMalloc(&d_mat, 9 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vec, 3 * sizeof(double)));

    // 2) Simulation des trajectoires
    int blockSize = 256;
    int gridPaths = (N_paths + blockSize - 1) / blockSize;

    simulatePathsKernel<<<gridPaths, blockSize>>>(
        d_paths, S0, r, sigma, T, N_steps, N_paths, 123456ULL);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 3) Payoff à maturité
    initCashflowsKernel<<<gridPaths, blockSize>>>(
        d_paths, d_cf_next, K, N_steps, N_paths);
    CUDA_CHECK(cudaDeviceSynchronize());

    double dt = T / (double)(N_steps - 1);
    double discount = exp(-r * dt);

    // 4) Backward LSMC
    for (int t = N_steps - 2; t >= 0; --t) {
        // Reset mat, vec
        CUDA_CHECK(cudaMemset(d_mat, 0, 9 * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_vec, 0, 3 * sizeof(double)));

        // Construire A^T A et A^T y
        buildNormalEquationsKernel<<<gridPaths, blockSize>>>(
            d_paths, d_cf_next, K, discount, t, N_paths, d_mat, d_vec);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Mettre à jour les cashflows à la date t
        updateCashflowsKernel<<<gridPaths, blockSize>>>(
            d_paths, d_cf_next, d_cf_curr,
            K, discount, t, N_paths, d_mat, d_vec);
        CUDA_CHECK(cudaDeviceSynchronize());

        // swap cf_next / cf_curr
        std::swap(d_cf_next, d_cf_curr);
    }

    // 5) Moyenne des cashflows à t=0 pour obtenir le prix
    std::vector<double> h_cf0(N_paths);
    CUDA_CHECK(cudaMemcpy(h_cf0.data(), d_cf_next, size_vec, cudaMemcpyDeviceToHost));

    double sum = 0.0;
    for (int i = 0; i < N_paths; ++i) sum += h_cf0[i];
    double price = sum / (double)N_paths;

    // 6) Nettoyage
    CUDA_CHECK(cudaFree(d_paths));
    CUDA_CHECK(cudaFree(d_cf_curr));
    CUDA_CHECK(cudaFree(d_cf_next));
    CUDA_CHECK(cudaFree(d_mat));
    CUDA_CHECK(cudaFree(d_vec));

    return price;
}

} // namespace LSMC_GPU
