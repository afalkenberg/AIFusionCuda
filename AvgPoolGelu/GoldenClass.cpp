#include "AvgPoolGelu.cuh"
#include "AvgPool.cuh"
#include "Gelu.cuh"
#include "GoldenClass.h"
#include<iostream> 

void GoldenClass::setInputs(std::vector<std::vector<float>>* inp, int k_size_i, int k_size_j, int st_i, int st_j) {
    input = inp; 
    kernel_size_i = k_size_i;
    kernel_size_j = k_size_j;
    stride_i = st_i; 
    stride_j = st_j;
    input_height = input->size();
    input_width = (*input)[0].size();
    output_height = (input_height - kernel_size_j) / stride_j + 1;
    output_width = (input_width - kernel_size_i) / stride_i + 1;
}

void GoldenClass::setInputs(std::vector<std::vector<int8_t>>* inp, int k_size_i, int k_size_j, int st_i, int st_j) {
    kernel_size_i = k_size_i;
    kernel_size_j = k_size_j;
    stride_i = st_i;
    stride_j = st_j;
    input_height = inp->size();
    input_width = (*inp)[0].size();
    output_height = (input_height - kernel_size_j) / stride_j + 1;
    output_width = (input_width - kernel_size_i) / stride_i + 1;
    input = new std::vector<std::vector<float>>;
    for (int i = 0; i < input_height; i++) {
        std::vector<float> finp;
        for (int j = 0; j < input_width; j++) {
            float val = (*inp)[i][j] / 128.0f;
            finp.push_back(val);
        }
        input->push_back(finp);
    }
}

void GoldenClass::AvgPool2d() {
    std::vector<float> temp(output_width); // associated with i
    output = new std::vector<std::vector<float>*>(output_height);  /// j 

    for (int j = 0; j < output_height; j++) {
        (*output)[j] = new std::vector<float>(output_width);
    }

    for (int j = 0; j < output_height; ++j) {
        for (int i = 0; i < output_width; ++i) {
            float sum = 0;
            for (int ki = 0; ki < kernel_size_i; ++ki) {
                for (int kj = 0; kj < kernel_size_j; ++kj) {
                    std::vector<float> outer = (*input)[j * stride_j + kj];
                    sum += outer[i * stride_i + ki];
                }
            }
            (*(*output)[j])[i] = sum  / (kernel_size_i * kernel_size_j);
        }
    }
}

void GoldenClass::Gelu() {
    for (int i = 0; i < output_height; ++i) {
        for (int j = 0; j < output_width; ++j) {
            float val = (*(*output)[i])[j];
            float sqrt2Divpi = 0.7978845608028654;
            (*(*output)[i])[j] = 0.5f * val * (1 + tanh(sqrt2Divpi * (val + 0.044715 * (val * val * val))));
        }
    }
}

void GoldenClass::makeCudaInput() {
    cudaInput = new float[input_height * input_width];
    for (int j = 0; j < input_height; ++j) {
        for (int i = 0; i < input_width; ++i) {
            std::vector<float> outer = (*input)[j];
            float val = outer[i];
            cudaInput[i + j * input_width] = val;
        }
    }
}


cudaError_t cudaStatus;

int GoldenClass::runAvgPoolGeluCuda() {
    // { 6, 4, kernel_size_i, kernel_size_j, stride_i, stride_j };
    b[0] = input_width;
    b[1] = input_height;
    b[2] = kernel_size_i;
    b[3] = kernel_size_j;
    b[4] = stride_i;
    b[5] = stride_j;

    c = new float[output_height * output_width];

    cudaStatus = avgGeluWithCuda(c, cudaInput, b, input_height * input_width, getOutputWidth(), getOutputHeight());
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "avgGeluWithCuda failed!");
        return 1;
    }

    return 0;
}


int GoldenClass::runAvgPoolCuda() {
    // { 6, 4, kernel_size_i, kernel_size_j, stride_i, stride_j };
    b[0] = input_width;
    b[1] = input_height;
    b[2] = kernel_size_i;
    b[3] = kernel_size_j;
    b[4] = stride_i;
    b[5] = stride_j;

    avgOut = new float[output_height * output_width];
    // d = new float[output_height * output_width];

    cudaStatus = avgPoolWithCuda(avgOut, cudaInput, b, input_height * input_width, getOutputWidth(), getOutputHeight());
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "avgGeluWithCuda failed!");
        return 1;
    }

//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }

    return 0;
}



int GoldenClass::runGeluCuda() {
    // { 6, 4, kernel_size_i, kernel_size_j, stride_i, stride_j };
    b[0] = output_width;
    b[1] = output_height;

    c = new float[output_height * output_width];

    cudaStatus = geluWithCuda<float>(c, avgOut, b, output_width, output_height);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "avgGeluWithCuda failed!");
        return 1;
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}


int GoldenClass::closeCuda() {
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}


void GoldenClass::makeCudaOutput() {
    std::vector<float> temp(output_width); // associated with i
    cudaOutput = new std::vector<std::vector<float>*>(output_height);  /// j 
    for (int j = 0; j < output_height; j++) {
        (*cudaOutput)[j] = new std::vector<float>(output_width);
    }

    for (int j = 0; j < output_height; ++j) {
        for (int i = 0; i < output_width; ++i) {
            (*(*cudaOutput)[j])[i] = c[i + j * getOutputWidth()];
        }
    }
}


std::vector<std::vector<float>*>* GoldenClass::getOutput() {
    return output; 
}

std::vector<std::vector<float>*>* GoldenClass::getCudaOutput() {
    return cudaOutput;
}

void GoldenClass::print() {
    for (const auto& row : *output) {
        for (float val : *row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

void GoldenClass::printCuda() {
    std::cout << output_height << " output_height " << cudaOutput->size() << std::endl;
    
    std::cout << output_width << " output_width " << (*cudaOutput)[0]->size() << std::endl;


    for (const auto& row : *cudaOutput) {
        for (float val : *row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

unsigned int GoldenClass::getOutputWidth() {
    return output_width;
}

unsigned int GoldenClass::getOutputHeight() {
    return output_height;
}

int GoldenClass::countErrors()  {
    float eps = 0.0000001;
    int cnt = 0;
    for (int j = 0; j < output_height; ++j) {
        for (int i = 0; i < output_width; ++i) {
            if (abs ((*(*cudaOutput)[j])[i] - (*(*output)[j])[i]) > eps) {
                cnt++;
           }
        }
    }
    return cnt;
}
