#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_functions.h>

#include "lsmc.hpp"

// ======================================================
// Helpers CUDA
// ======================================================

inline void gpuAssert(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA ERROR %d (%s) at %s:%d\n",
            (int)code, cudaGetErrorString(code), file, line);
        abort();
    }
}
#define CUDA_CHECK(x) gpuAssert((x), __FILE__, __LINE__)

// ======================================================
// Indexation path-major
// ======================================================
__device__ __forceinline__
int idx_path_time(int i, int t, int N_steps)
{
    return i * (N_steps + 1) + t;
}

// ======================================================
// PAYOFF KERNEL
// ======================================================
__global__
void payoff_kernel(const float* d_paths, float* d_payoff,
    float K, int N_steps, int N_paths)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N_paths * (N_steps + 1);
    if (id >= total) return;

    float S = d_paths[id];
    d_payoff[id] = fmaxf(K - S, 0.0f);
}

// ======================================================
// INIT CASHFLOWS
// ======================================================
__global__
void init_cashflows_kernel(const float* d_payoff,
    float* d_cashflows,
    int N_steps, int N_paths)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_paths) return;

    int idxT = i * (N_steps + 1) + N_steps;
    d_cashflows[i] = d_payoff[idxT];
}

// ======================================================
// REGRESSION GPU (simple atomicAdd version)
// ======================================================
__global__
void regression_sums_kernel(const float* __restrict__ d_paths,
    const float* __restrict__ d_payoff,
    const float* __restrict__ d_cashflows,
    int t,
    int N_steps,
    int N_paths,
    float discount,
    double* __restrict__ d_sums)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < N_paths; i += stride)
    {
        int id_xt = idx_path_time(i, t, N_steps);

        double immediate = (double)d_payoff[id_xt];
        if (immediate <= 0.0) continue;

        double S = (double)d_paths[id_xt];
        double Y = (double)d_cashflows[i] * (double)discount;

        double phi0 = 1.0;
        double phi1 = S;
        double phi2 = S * S;

        // A^T A
        atomicAdd(&d_sums[0], phi0 * phi0);
        atomicAdd(&d_sums[1], phi0 * phi1);
        atomicAdd(&d_sums[2], phi0 * phi2);
        atomicAdd(&d_sums[3], phi1 * phi1);
        atomicAdd(&d_sums[4], phi1 * phi2);
        atomicAdd(&d_sums[5], phi2 * phi2);

        // A^T y
        atomicAdd(&d_sums[6], phi0 * Y);
        atomicAdd(&d_sums[7], phi1 * Y);
        atomicAdd(&d_sums[8], phi2 * Y);
    }
}

void computeRegressionSumsGPU(const float* d_paths,
    const float* d_payoff,
    const float* d_cashflows,
    int t,
    int N_steps,
    int N_paths,
    float discount,
    RegressionSumsGPU& out)
{
    double* d_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sums, 9 * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sums, 0, 9 * sizeof(double)));

    int block = 256;
    int grid = (N_paths + block - 1) / block;

    regression_sums_kernel << <grid, block >> > (
        d_paths, d_payoff, d_cashflows,
        t, N_steps, N_paths, discount,
        d_sums
        );
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    double h_sums[9];
    CUDA_CHECK(cudaMemcpy(h_sums, d_sums, 9 * sizeof(double),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_sums));

    out.a00 = h_sums[0];
    out.a01 = h_sums[1];
    out.a02 = h_sums[2];
    out.a11 = h_sums[3];
    out.a12 = h_sums[4];
    out.a22 = h_sums[5];
    out.b0 = h_sums[6];
    out.b1 = h_sums[7];
    out.b2 = h_sums[8];
}

// ======================================================
// UPDATE CASHFLOWS
// ======================================================
__global__
void update_cashflows_kernel(const float* d_paths,
    const float* d_payoff,
    float* d_cashflows,
    BetaGPU beta,
    float discount,
    int t, int N_steps, int N_paths)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_paths) return;

    int id = idx_path_time(i, t, N_steps);
    float immediate = d_payoff[id];
    float S = d_paths[id];

    if (immediate > 0.0f)
    {
        double cont = beta.beta0 +
            beta.beta1 * S +
            beta.beta2 * S * S;

        if ((double)immediate > cont)
            d_cashflows[i] = immediate;
        else
            d_cashflows[i] *= discount;
    }
    else {
        d_cashflows[i] *= discount;
    }
}

void updateCashflowsGPU(const float* d_paths,
    const float* d_payoff,
    float* d_cashflows,
    const BetaGPU& beta,
    float discount,
    int t,
    int N_steps,
    int N_paths)
{
    int block = 256;
    int grid = (N_paths + block - 1) / block;

    update_cashflows_kernel <<< grid, block >>> (
        d_paths, d_payoff, d_cashflows,
        beta, discount,
        t, N_steps, N_paths
        );
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

