

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

cudaError_t avgPoolWithCuda(float* c, float* a, int* b, unsigned int size, unsigned int output_size_i, unsigned int output_size_j);

