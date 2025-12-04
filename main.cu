#include <iostream>
#include <cuda_runtime.h>

// ======================================================
//  Vérification des erreurs CUDA
// ======================================================
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess)
    {
        std::cerr << "CUDA ERROR : " << cudaGetErrorString(code)
            << "  (" << code << ")"
            << "  at " << file << ":" << line << std::endl;
    }
}

// ======================================================
//  Kernel CUDA simple
// ======================================================
__global__ void kernel_test(float* d_out)
{
    int i = threadIdx.x;
    d_out[i] = i * 2.0f;
}

// ======================================================
//  MAIN : test complet CUDA
// ======================================================
int main()
{
    std::cout << "=== TEST CUDA ===" << std::endl;

    // ---------- Vérifier disponibilité CUDA ----------
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        std::cerr << "AUCUN GPU CUDA DISPONIBLE !" << std::endl;
        return -1;
    }

    std::cout << "Nombre de GPU détectés : " << deviceCount << std::endl;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "GPU 0 : " << prop.name << std::endl;
    std::cout << "Compute Capability : " << prop.major << "." << prop.minor << std::endl;

    // ---------- Allocation GPU ----------
    float* d_out;
    int N = 8;

    CUDA_CHECK(cudaMalloc((void**)&d_out, N * sizeof(float)));

    // ---------- Lancement kernel ----------
    kernel_test << <1, N >> > (d_out);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---------- Récupération des résultats ----------
    float h_out[8] = { 0 };
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    // ---------- Affichage ----------
    std::cout << "Résultats kernel :" << std::endl;
    for (int i = 0; i < N; i++)
        std::cout << "h_out[" << i << "] = " << h_out[i] << std::endl;

    // ---------- Nettoyage ----------
    CUDA_CHECK(cudaFree(d_out));

    std::cout << "=== CUDA OK ===" << std::endl;
    return 0;
}

