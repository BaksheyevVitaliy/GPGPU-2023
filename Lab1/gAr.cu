
#pragma once
#include <device_launch_parameters.h>
#include <cstring>
#include <iostream>

#include "cuda_runtime.h"
#include "gAr.cuh"

gAr::gAr(int ndim, const int* shape, const double* postData)
{
	this->dimNumb = ndim;
	this->dimVal = new int[ndim];
	std::memcpy(this->dimVal, shape, ndim * sizeof(int));
	this->elemNumb = 1;
	for (int i = 0; i < ndim; i++) {
		this->elemNumb *= shape[i];
	}
	cudaMalloc(&this->data, this->elemNumb * sizeof(double));
	fill(postData);
}

void gAr::fill(const double* moreValue)
{
	cudaMemcpy(this->data, moreValue, this->elemNumb * sizeof(double), cudaMemcpyHostToDevice);
}

gAr gAr::operator+=(const gAr* B)
{
	for (int i = 0; i < this->elemNumb; i++) {
		this->data[i] += B->data[i];
	}
	return *this;
}

gAr gAr::operator+(const gAr* B)
{
	for (int i = 0; i < this->elemNumb; i++) {
		this->data[i] += B->data[i];
	}
	return *this;
}

__global__ void SumKernel(double* a, double* b, int n) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < n) {
		a[idx] += b[idx];
	}
}

void gAr::plus(const gAr* B)
{
	dim3 blocks(1+ (this->elemNumb - 1) / 1024, 1, 1);
	dim3 threads(std::min(this->elemNumb, 1024), 1, 1);
	SumKernel <<<blocks, threads >>> (this->data, B->data, this->elemNumb);
	cudaDeviceSynchronize();
}

void gAr::PrintReccur(int dimention, double* data)
{
	if (dimention == 0)
		this->currElem = 0;
	if (this->dimNumb - dimention == 1) {
		std::cout << data[this->currElem];
		this->currElem++;
		for (int i = 1; i < this->dimVal[dimention]; i++) {
			std::cout << ", " << data[this->currElem];
			currElem++;
		}
		return;
	}
	for (int i = 0; i < this->dimVal[dimention]; i++) {
		if (this->dimNumb - dimention > 2)
			std::cout << "[";
		this->Print(dimention + 1);
		if (this->dimNumb - dimention == 2)
			std::cout << ";";
		if (this->dimNumb - dimention > 2)
			std::cout << "]";
		if (i != this->dimVal[dimention] - 1)
			std::cout << "\n";
	}
	if (dimention == 0)
		std::cout << "\n";
}
void gAr::Print(int dimention) {
	double* buf = new double[this->elemNumb];
	cudaMemcpy(buf, this->data, this->elemNumb * sizeof(double), cudaMemcpyDeviceToHost);
	PrintReccur(dimention, buf);
}

gAr::~gAr()
{
	cudaFree(this->data);
	delete[] this->dimVal;
}
