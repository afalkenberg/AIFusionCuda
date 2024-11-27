#include <iostream>
#include "GoldenClass.h"
#include <ctime>


int cudaCall(int dim_i, int dim_j, bool merged)
{

    // defining the inputs and golden //

    std::vector<std::vector<float>> input; 
    
    for (int i = 0; i < dim_i; i++) {
        std::vector<float> inp; 
        for (int j = 0; j < dim_j; j++) {
            inp.push_back(rand()/32000.0f);
        }
        input.push_back(inp);
    }

    int kernel_size_i = 8;
    int kernel_size_j = 8;

    int stride_i = 8;
    int stride_j = 8;

    GoldenClass GC;
    GC.setInputs(&input, kernel_size_i, kernel_size_j, stride_i, stride_j);
    clock_t startAvgPool; 
    clock_t startGelu;
    clock_t endGelu;
    double avgTime;
    double geluTime;
    double avgGeluTime;

    startAvgPool = clock();
    GC.AvgPool2d();
    startGelu = clock(); 
   
    GC.Gelu();
    endGelu = clock();
    avgTime = ((double)(startGelu - startAvgPool)) / CLOCKS_PER_SEC;
    geluTime = ((double)(endGelu - startGelu)) / CLOCKS_PER_SEC;
    avgGeluTime = ((double)(endGelu - startAvgPool)) / CLOCKS_PER_SEC;

    std::cout << " AVG  TIME " << avgTime << std::endl; 
    std::cout << " GELU TIME " << geluTime << std::endl;
    std::cout << " BOTH TIME " << avgGeluTime << std::endl;

    // GC.print();
    // std::cout << " -------------- " << std::endl;

    GC.makeCudaInput(); 

    clock_t cudaStart;
    clock_t cudaGeluStart;
    clock_t cudaEnd;
    double cudaTotalTime;
    double cudaAvgTime;
    double cudaGeluTime;

    cudaStart = clock();
    if (merged == false) {
        GC.runAvgPoolCuda();
        cudaGeluStart = clock();
        GC.runGeluCuda();
    }
    else {
        GC.runAvgPoolGeluCuda();
    }

    cudaEnd = clock(); 
    GC.makeCudaOutput(); 
    //GC.printCuda();

    if (merged == false) {
        cudaAvgTime = ((double)(cudaGeluStart - cudaStart)) / CLOCKS_PER_SEC;
        cudaGeluTime = ((double)(cudaEnd - cudaGeluStart)) / CLOCKS_PER_SEC;
    }
    cudaTotalTime = ((double)(cudaEnd - cudaStart)) / CLOCKS_PER_SEC;

    if (merged == false) {
        std::cout << " CUDA AVG   TIME " << cudaAvgTime << std::endl;
        std::cout << " CUDA GEL   TIME " << cudaGeluTime << std::endl;
    }

    std::cout << " CUDA TOTAL TIME " << cudaTotalTime << std::endl;
    std::cout << " compare " << GC.countErrors() << std::endl;

    GC.closeCuda(); 
    std::cout << dim_i << " " << cudaTotalTime << std::endl;
    return 0;
}

int main(int argc, char** arg)
{
    std::vector<int> dim_i = { 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944, 3072};
    std::vector<int> dim_j = { 512, 1024, 1536, 2048, 2560, 3072 };

    for (int dj : dim_j) {
        std::cout << " ____not merged ____________ " << dj << " ------------ " << std::endl;
        for (int di : dim_i) {
            cudaCall(di, dj, false);
        }
        std::cout << std::endl;
        std::cout << " _______ merged ____________ " << dj << " ------------ " << std::endl;
        for (int di : dim_i) {
            cudaCall(di, dj, true);
        }
        std::cout << std::endl;
    }
    return 0;
}
