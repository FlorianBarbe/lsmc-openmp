#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void testKernel(int* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = i;
}

extern "C" void runTestKernel()
{
    int* d_out;
    int N = 32;

    cudaMalloc(&d_out, N * sizeof(int));
    testKernel << <1, N >> > (d_out);
    cudaDeviceSynchronize();
    cudaFree(d_out);
}
