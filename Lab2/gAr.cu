#define CUDACC

#include <device_launch_parameters.h>
#include <cstring>
#include <iostream>
#include<math.h>
#include <cooperative_groups.h>

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
	if (moreValue != nullptr) {
		cudaMemcpy(this->data, moreValue, this->elemNumb * sizeof(double), cudaMemcpyHostToDevice);
	}
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

__global__ void plusParallel(double* A, double* B, int eNumb) {
	int thrN = blockDim.x * blockIdx.x + threadIdx.x;
	if (thrN < eNumb) {
		A[thrN] += B[thrN];
	}
}

void gAr::plus(const gAr* B)
{
	dim3 blocks(1+ (this->elemNumb - 1) / 1024, 1, 1);
	dim3 threads(std::min(this->elemNumb, 1024), 1, 1);
	plusParallel <<<blocks, threads>>> (this->data, B->data, this->elemNumb);
	cudaDeviceSynchronize();
}

double gAr::error(const gAr* B)
{
	double maxError = 0;
	double buf;
	double* bufArray1 = new double[this->elemNumb];
	cudaMemcpy(bufArray1, this->data, this->elemNumb * sizeof(double), cudaMemcpyDeviceToHost);
	double* bufArray2 = new double[this->elemNumb];
	cudaMemcpy(bufArray2, B->data, this->elemNumb * sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < this->elemNumb; i++) {
		buf = abs(bufArray1[i] - bufArray2[i]);
		if (buf > maxError)
			maxError = buf;
	}
	return maxError;
}

double gAr::error(const cAr* B)
{
	double maxError = 0;
	double buf;
	double* bufArray = new double[this->elemNumb];
	cudaMemcpy(bufArray, this->data, this->elemNumb * sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < this->elemNumb; i++) {
		buf = abs(bufArray[i] - B->data[i]);
		if (buf > maxError)
			maxError = buf;
	}
	return maxError;
}

__global__ void multParallel(double* C, double* A, double* B, int eNumb, int AValRow, int AValCol, int BValCol) {
	int thrN = blockDim.x * blockIdx.x + threadIdx.x;
	if (thrN < eNumb) {
		int i = thrN / AValRow;
		int j = thrN - i * BValCol;
		double buf = 0;
		for (int k = 0; k < AValCol; k++) {
			buf += A[i * AValCol + k] * B[k * BValCol + j];
		}
		C[i * BValCol + j] = buf;
	}
}

void gAr::mult(const gAr* A, const gAr* B)
{
	dim3 blocks(1 + (this->elemNumb - 1) / 1024, 1, 1);
	dim3 threads(std::min(this->elemNumb, 1024), 1, 1);
	int AValRow = A->dimVal[0], AValCol = A->dimVal[1], BValCol = B->dimVal[1];
	multParallel <<<blocks, threads >>> (this->data, A->data,  B->data, this->elemNumb, AValRow, AValCol, BValCol);
	cudaDeviceSynchronize();
}

__global__ void shMultSquareParallel(double* C, double* A, double* B, int eNumb, int AValRow, int AValCol, int BValCol) {
	__shared__ double ASquare[32][32];
	__shared__ double BSquare[32][32];

	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	double buf = 0;

	for (int l = 0; l < 1 + (AValCol - 1) / 32; l++) {
		if (i < AValRow && (threadIdx.x + l * 32) < AValCol) {
			ASquare[threadIdx.y][threadIdx.x] = A[i * AValCol + threadIdx.x + l * 32];
		}
		else {
			ASquare[threadIdx.y][threadIdx.x] = 0;
		}
		if (j < BValCol && (threadIdx.y + l * 32) < AValCol) {
			BSquare[threadIdx.y][threadIdx.x] = B[(threadIdx.y + l * 32) * BValCol + j];
		}
		else {
			BSquare[threadIdx.y][threadIdx.x] = 0;
		}
		__syncthreads();

		for (size_t k = 0; k < 32; k++) {
			buf += ASquare[threadIdx.y][k] * BSquare[k][threadIdx.x];
		}
		__syncthreads();
	}
	if (i < AValRow && j < BValCol) {
		C[i * BValCol + j] = buf;
	}
}

void gAr::shMultSquare(const gAr* A, const gAr* B)
{
	dim3 blocksMatrix(1 + (A->dimVal[0] - 1) / 32, 1 + (B->dimVal[1] - 1) / 32, 1);
	dim3 threadsMatrix(32, 32, 1);
	int AValRow = A->dimVal[0], AValCol = A->dimVal[1], BValCol = B->dimVal[1];
	shMultSquareParallel <<<blocksMatrix, threadsMatrix >>> (this->data, A->data, B->data, this->elemNumb, AValRow, AValCol, BValCol);
	cudaDeviceSynchronize();
}

__global__ void shMultRowParallel(double* C, double* A, double* B, int eNumb, int AValRow, int AValCol, int BValCol) {
	__shared__ double Arow[1024];
	double buf = 0;
	int i = blockIdx.y;
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	for (int l = 0; l < 1 + (AValCol - 1) / 1024; l++) {
		if (l * 1024 + threadIdx.x < AValCol) {
			Arow[threadIdx.x] = A[AValCol * i + l * 1024 + threadIdx.x];
		}
		else {
			Arow[threadIdx.x] = 0;
		}
		__syncthreads();
		for (int k = 0; k < 1024; k++) {
			if (k + l * 1024 < AValCol) {
				buf += Arow[k] * B[(k + l * 1024) * BValCol + j];
			}
			else
				break;
		}
		__syncthreads();
	}
	if ((i < AValRow) && (j < BValCol)) {
		C[i * BValCol + j] = buf;
	}
}

void gAr::shMultRow(const gAr* A, const gAr* B)
{
	dim3 blocksMatrix( 1 + (B->dimVal[1] - 1) / 1024,A->dimVal[0], 1);
	dim3 threadsMatrix(1024, 1, 1);
	int AValRow = A->dimVal[0], AValCol = A->dimVal[1], BValCol = B->dimVal[1];
	shMultRowParallel <<<blocksMatrix, threadsMatrix >>> (this->data, A->data, B->data, this->elemNumb, AValRow, AValCol, BValCol);
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
