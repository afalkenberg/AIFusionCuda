
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_fp16.h>

template<typename T>
cudaError_t geluWithCuda(T* c, T* a, int* b, unsigned int output_size_i, unsigned int output_size_j);

