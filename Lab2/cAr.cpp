#include <cstring>
#include <iostream>

#include "cAr.h"

cAr::cAr(int ndim, const int* shape, const double* postData)
{
	this->dimNumb = ndim;
	this->dimVal = new int[ndim];
	std::memcpy(this->dimVal, shape, ndim * sizeof(int));
	this->elemNumb = 1;
	for (int i = 0; i < ndim; i++) {
		this->elemNumb *= shape[i];
	}
	this->data = new double[this->elemNumb];
	fill(postData);
}

void cAr::fill(const double* moreValue)
{
	if (moreValue != nullptr) {
		std::memcpy(this->data, moreValue, this->elemNumb * sizeof(double));
	}
}

cAr cAr::operator+=(const cAr* B)
{
	for (int i = 0; i < this->elemNumb; i++) {
		this->data[i] += B->data[i];
	}
	return *this;
}

cAr cAr::operator+(const cAr* B)
{
	for (int i = 0; i < this->elemNumb; i++) {
		this->data[i] += B->data[i];
	}
	return *this;
}

void cAr::plus(const cAr* B)
{
	for (int i = 0; i < this->elemNumb; i++) {
		this->data[i] += B->data[i];
	}
}
double cAr::error(const cAr* B)
{
	double maxError = 0;
	double buf;
	for (int i = 0; i < this->elemNumb; i++) {
		buf = abs(this->data[i] - B->data[i]);
		if (buf > maxError)
			maxError = buf;
	}
	return maxError;
}


void cAr::mult(const cAr* A, const cAr* B)
{
	this->currElem = 0;
	int AValRow = A->dimVal[0], AValCol = A->dimVal[1], BValCol = B->dimVal[1];
	double buf;
	for (int i = 0; i < AValRow; i++) {
		for (int j = 0; j < BValCol; j++) {
			buf = 0;
			for (int k = 0; k < AValCol; k++) {
				buf += A->data[i * AValCol + k] * B->data[k * BValCol + j];
			}
			this->data[this->currElem] = buf;
			this->currElem++;
		}
	}
}

void cAr::Print(int dimention)
{
	if (dimention == 0)
		this->currElem = 0;
	if (this->dimNumb - dimention == 1) {
		std::cout << this->data[this->currElem];
		this->currElem++;
		for (int i = 1; i < this->dimVal[dimention]; i++) {
			std::cout << ", "  <<  this->data[this->currElem];
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

cAr::~cAr()
{
	delete[] this->data;
	delete[] this->dimVal;
}
