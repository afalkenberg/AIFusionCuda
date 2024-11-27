#pragma once

#include <vector>


class GoldenClass
{
private:

	std::vector<std::vector<float>>* input;
	std::vector<std::vector<float>*>* output;

	float* cudaInput;
	std::vector<std::vector<float>*>* cudaOutput;

	int kernel_size_i;
	int kernel_size_j;
	int stride_i;
	int stride_j;

	int input_height; 
	int input_width;
	int output_height;
	int output_width;

	int b[6]; // = { 6, 4, kernel_size_i, kernel_size_j, stride_i, stride_j };
	float* avgOut;
	float* geluOut;
	float* c;

	
public:

	void setInputs(std::vector<std::vector<float>>* inp, int k_size_i, int k_size_j, int st_i, int st_j);

	void AvgPool2d();
    void Gelu();

	std::vector<std::vector<float>*>* getOutput();
	std::vector<std::vector<float>*>* getCudaOutput();

	void makeCudaInput();
	void makeCudaOutput();
	int runAvgPoolGeluCuda();
	int runAvgPoolCuda();
	int runGeluCuda();

	int closeCuda(); 

	void print(); 
	void printCuda();

	unsigned int getOutputWidth(); 
	unsigned int getOutputHeight();
	int countErrors();

};

