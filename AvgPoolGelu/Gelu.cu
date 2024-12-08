#include "Gelu.cuh"
#include <stdio.h>
#include <iostream>

template <typename T> 
__global__ void geluKernel(T* c, const T* a, const int* b)
{
    T par2 = (T)0.5f;
    T par1 = (T)0.044715f;
    T sqrt2Divpi = (T)0.7978845608028654f;
    T one = (T)1.0f;

    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int matrixSize_i = b[0];
    int matrixSize_j = b[1];
    T sum = a[i + matrixSize_i * j];
    T x0 = sqrt2Divpi * (sum + par1 * (sum * sum * sum));
    T tanVal;
    tanVal = (T)tanhf(x0);
    c[i + matrixSize_i * j] = par2 * sum * (one + tanVal);
}


template <typename T>
cudaError_t geluWithCuda<T>(T* c, T* a, int* b, unsigned int output_size_i, unsigned int output_size_j)
{
    T* dev_a = 0;
    int* dev_b = 0;
    T* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, output_size_i * output_size_j * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, output_size_i * output_size_j * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, 6 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, output_size_i * output_size_j * sizeof(T), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, 6 * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    // avgGeluKernel<<<dim3(1,1), dim3(output_size_i, output_size_j)>>>(dev_c, dev_a, dev_b);

    geluKernel<T> <<<dim3(output_size_i, output_size_j), dim3(1, 1) >>> (dev_c, dev_a, dev_b);


    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "avgPoolKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching avgGeluKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, output_size_i * output_size_j * sizeof(T), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

template cudaError_t geluWithCuda<float>(float* c, float* a, int* b, unsigned int output_size_i, unsigned int output_size_j);
template cudaError_t geluWithCuda<double>(double* c, double* a, int* b, unsigned int output_size_i, unsigned int output_size_j);
template cudaError_t geluWithCuda<half>(half* c, half* a, int* b, unsigned int output_size_i, unsigned int output_size_j);

