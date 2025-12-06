// gbm_cuda.cu
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>

#include "gbm.hpp"

#ifndef __CUDACC__
#error "gbm.cu must be compiled with NVCC (CUDA compiler)."
#endif


// =====================================================================================
// Helpers CUDA
// =====================================================================================
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA ERROR %d (%s) at %s:%d\n",
            code, cudaGetErrorString(code), file, line);
    }
}

#define CURAND_CHECK(ans) { curandAssert((ans), __FILE__, __LINE__); }
inline void curandAssert(curandStatus_t code, const char* file, int line)
{
    if (code != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "CURAND ERROR %d at %s:%d\n", code, file, line);
    }
}

// Layout mémoire coalescé
__device__ __forceinline__
int idx_tp(int t, int path, int nPaths)
{
    return t * nPaths + path;
}

// =====================================================================================
// 1) Kernel Philox
// =====================================================================================
__global__
void gbm_paths_philox_kernel(GbmParams params,
    unsigned long long seed,
    float* __restrict__ d_paths)
{
    int pathId = blockIdx.x * blockDim.x + threadIdx.x;
    if (pathId >= params.nPaths) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, pathId, 0ULL, &state);

    float dt = params.T / params.nSteps;
    float drift = (params.r - 0.5f * params.sigma * params.sigma) * dt;
    float vol = params.sigma * sqrtf(dt);

    float S = params.S0;
    d_paths[idx_tp(0, pathId, params.nPaths)] = S;

    for (int t = 1; t <= params.nSteps; ++t) {
        float z = curand_normal(&state);
        S *= expf(drift + vol * z);
        d_paths[idx_tp(t, pathId, params.nPaths)] = S;
    }
}

// =====================================================================================
// 2) Kernel Sobol
// =====================================================================================
__global__
void gbm_paths_from_normals_kernel(GbmParams params,
    const float* __restrict__ d_normals,
    float* __restrict__ d_paths)
{
    int pathId = blockIdx.x * blockDim.x + threadIdx.x;
    if (pathId >= params.nPaths) return;

    float dt = params.T / params.nSteps;
    float drift = (params.r - 0.5f * params.sigma * params.sigma) * dt;
    float vol = params.sigma * sqrtf(dt);

    float S = params.S0;
    d_paths[idx_tp(0, pathId, params.nPaths)] = S;

    for (int t = 1; t <= params.nSteps; ++t) {
        float z = d_normals[idx_tp(t - 1, pathId, params.nPaths)];
        S *= expf(drift + vol * z);
        d_paths[idx_tp(t, pathId, params.nPaths)] = S;
    }
}

// =====================================================================================
// 3) Implémentation host
// =====================================================================================
void simulate_gbm_paths_cuda(const GbmParams& params,
    RNGType rng,
    float* d_paths,
    unsigned long long seed,
    cudaStream_t stream)
{
    int nPaths = params.nPaths;
    int nSteps = params.nSteps;

    dim3 block(256);
    dim3 grid((nPaths + block.x - 1) / block.x);

    if (rng == RNGType::PseudoPhilox) {
        gbm_paths_philox_kernel << <grid, block, 0, stream >> > (params, seed, d_paths);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    else {
        size_t nDim = static_cast<size_t>(nSteps);
        size_t total = static_cast<size_t>(nPaths) * nSteps;

        float* d_normals = nullptr;
        CUDA_CHECK(cudaMalloc(&d_normals, total * sizeof(float)));

        curandGenerator_t gen;
        CURAND_CHECK(curandCreateGenerator(&gen,
            CURAND_RNG_QUASI_SCRAMBLED_SOBOL32));

        CURAND_CHECK(curandSetQuasiRandomGeneratorDimensions(gen, nDim));
        CURAND_CHECK(curandSetStream(gen, stream));

        CURAND_CHECK(curandGenerateNormal(gen, d_normals,
            total, 0.0f, 1.0f));
        CURAND_CHECK(curandDestroyGenerator(gen));

        gbm_paths_from_normals_kernel << <grid, block, 0, stream >> > (
            params, d_normals, d_paths);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaFree(d_normals));
    }
}
