#include <stdio.h>

__global__ void initialize(float *x, float a, float *y, float b, size_t n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        x[i] = a;
        y[i] = b;
    }
}

__global__ void saxpy(float a, float *x, float *y, size_t n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        y[i] = a * x[i] + y[i];
}

int main()
{
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);

    const size_t n = 1 << 24;

    float *x;
    float *y;
    cudaMallocManaged(&x, n * sizeof(float));
    cudaMallocManaged(&y, n * sizeof(float));

    size_t grid = 32 * properties.multiProcessorCount;
    size_t block = 256;

    initialize<<<grid, block>>>(x, 2, y, 3, n);

    saxpy<<<grid, block>>>(5, x, y, n);

    cudaDeviceSynchronize();

    cudaMemPrefetchAsync(y, n, cudaCpuDeviceId);

    printf("y[0] = %f\n", y[0]);
    printf("y[n - 1] = %f\n", y[n - 1]);

    return EXIT_SUCCESS;
}