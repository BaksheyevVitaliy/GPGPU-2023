#include <iostream>
#include <Windows.h>
#include <time.h>

#include"cAr.h"
#include"gAr.cuh"


void createRandArr(double* buffer, int elem_num) {
	const double a = 0;
	const double b = 1;
	for (int i = 0; i < elem_num; i++) {
		buffer[i] = a + ((double)std::rand() / RAND_MAX) * (b - a);
	}
}


int main(void) {
	int N = 1000;
	int Dim[2] = {N, N};
	int numElem = N * N;
	int NIter = 10;
	double err14 = 0, err12 = 0, err13 = 0;
	double* dataA = new double[numElem];
	double* dataB = new double[numElem];
	cAr* A1 = new cAr(2, Dim);
	cAr* B1 = new cAr(2, Dim);
	cAr* C1 = new cAr(2, Dim);
	gAr* A2 = new gAr(2, Dim);
	gAr* B2 = new gAr(2, Dim);
	gAr* C2 = new gAr(2, Dim);
	gAr* C3 = new gAr(2, Dim);
	gAr* C4 = new gAr(2, Dim);
	int64_t time_cpu = 0, time_gpu_row = 0, time_gpu_square = 0, time_gpu_naive = 0;
	LARGE_INTEGER frequency = { 0 }, time_stamp[2];
	if (!QueryPerformanceFrequency(&frequency)) {
		throw std::runtime_error("Performance counter error");
	}
	for (int i = 0; i < NIter; i++) {

		createRandArr(dataA, numElem);
		createRandArr(dataB, numElem);
		A1->fill(dataA);
		B1->fill(dataB);
		A2->fill(dataA);
		B2->fill(dataB);
		ZeroMemory(time_stamp, sizeof(time_stamp));
		QueryPerformanceCounter(time_stamp);
		C1->mult(A1, B1);
		QueryPerformanceCounter(time_stamp + 1);
		time_cpu += time_stamp[1].QuadPart - time_stamp[0].QuadPart;
		ZeroMemory(time_stamp, sizeof(time_stamp));
		QueryPerformanceCounter(time_stamp);
		C2->mult(A2, B2);
		QueryPerformanceCounter(time_stamp + 1);
		time_gpu_naive += time_stamp[1].QuadPart - time_stamp[0].QuadPart;
		ZeroMemory(time_stamp, sizeof(time_stamp));
		QueryPerformanceCounter(time_stamp);
		C3->shMultRow(A2, B2);
		QueryPerformanceCounter(time_stamp + 1);
		time_gpu_row += time_stamp[1].QuadPart - time_stamp[0].QuadPart;
		ZeroMemory(time_stamp, sizeof(time_stamp));
		QueryPerformanceCounter(time_stamp);
		C4->shMultSquare(A2, B2);
		QueryPerformanceCounter(time_stamp + 1);
		time_gpu_square += time_stamp[1].QuadPart - time_stamp[0].QuadPart;
		double buffErr14 = C4->error(C1);
		double buffErr12 = C2->error(C1);
		double buffErr13 = C3->error(C1);
		if (buffErr14 > err14)
			err14 = buffErr14;
		if (buffErr12 > err12)
			err12 = buffErr12;
		if (buffErr13 > err13)
			err13 = buffErr13;
		std::cout << i<<"\n";
	}
	std::cout << "Avg. CPU time: "
		<< ((long double)time_cpu /
			(frequency.QuadPart)) * 1000.0 / NIter
		<< " ms.\n";
	std::cout << "Avg. GPU non shared time: "
		<< ((long double)time_gpu_naive /
			(frequency.QuadPart)) * 1000.0 / NIter
		<< " ms.\n";
	std::cout << "Avg. GPU shared row time: "
		<< ((long double)time_gpu_row /
			(frequency.QuadPart)) * 1000.0 / NIter
		<< " ms.\n";
	std::cout << "Avg. GPU shared square time: "
		<< ((long double)time_gpu_square /
			(frequency.QuadPart)) * 1000.0 / NIter
		<< " ms.\n";
	std::cout << err12 << " " << err13 << " " << err14;
	delete A1;
	delete B1;
	delete C1;
	delete A2;
	delete B2;
	delete C2;
	delete C3;
	delete C4;
}