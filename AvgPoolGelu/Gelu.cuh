
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

cudaError_t geluWithCuda(float* c, float* a, int* b, unsigned int output_size_i, unsigned int output_size_j);

