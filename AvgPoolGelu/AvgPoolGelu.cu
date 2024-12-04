

#include "AvgPoolGelu.cuh"

#include <stdio.h>
#include <iostream>


__global__ void avgGeluKernel(float* c, const float* a, const int* b)
{
    // int i = threadIdx.x; // [0 1]  
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // int j = threadIdx.y; // [0 1]  

    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int matrixSize_i = b[0];
    int matrixSize_j = b[1];
    int kernelSize_i = b[2];
    int kernelSize_j = b[3];
    int stride_i = b[4]; 
    int stride_j = b[5];

    float sum = 0.0;
    for (int ki = 0; ki < kernelSize_i; ki++) {
        for (int kj = 0; kj < kernelSize_j; kj++) {
            sum = sum + a[stride_i * i + ki + kj * matrixSize_i + j * stride_j * matrixSize_i];
        }
    }

    int dj = (matrixSize_i - kernelSize_i) / stride_i + 1 ; 
    sum = sum / (kernelSize_i * kernelSize_j);
    float sqrt2Divpi = 0.7978845608028654;
    c[i + dj * j] = 0.5f * sum * (1 + tanhf(sqrt2Divpi * (sum + 0.044715 * (sum * sum * sum))));
}



// Helper function for using CUDA to avgGelu vectors in parallel.
cudaError_t avgGeluWithCuda(float *c, float *a, int *b, unsigned int inputSize, unsigned int output_size_i, unsigned int output_size_j)
{
    float *dev_a = 0;
    int *dev_b = 0;
    float *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, output_size_i * output_size_j * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, inputSize * sizeof(float));
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
    cudaStatus = cudaMemcpy(dev_a, a, inputSize * sizeof(float), cudaMemcpyHostToDevice);
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

//    avgGeluKernel <<<dim3(output_size_i/4, output_size_j/4), dim3(4, 4) >> > (dev_c, dev_a, dev_b);

    avgGeluKernel << <dim3(output_size_i, output_size_j), dim3(1,1) >> > (dev_c, dev_a, dev_b);


    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "avgGeluKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
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
    cudaStatus = cudaMemcpy(c, dev_c, output_size_i * output_size_j * sizeof(float), cudaMemcpyDeviceToHost);
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
