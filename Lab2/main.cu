#include <iostream>
#include <stdio.h>
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
	double changeN = 100;
	double koefN = 10;
	int NIterN = 2;
	int NIter = 10;
	FILE* fp;
	fp = fopen("GPGPUExperimentResults.txt", "w");
	for(int j = 0; j < NIterN; j++){
		int N = ceil(changeN);
		int Dim[2] = { N, N };
		int numElem = N * N;
		double* dataA = new double[numElem];
		double* dataB = new double[numElem];
		//for (int i = 0; i < numElem; i++) {
		//	dataA[i] = i;
		//}
		//for (int i = 0; i < numElem; i++) {
		//	dataB[i] = numElem - i;
		//}
		cAr* A1 = new cAr(2, Dim);
		cAr* B1 = new cAr(2, Dim);
		cAr* C1 = new cAr(2, Dim);
		gAr* A2 = new gAr(2, Dim);
		gAr* B2 = new gAr(2, Dim);
		gAr* C2 = new gAr(2, Dim);
		gAr* C3 = new gAr(2, Dim);
		gAr* C4 = new gAr(2, Dim);
		std::cout << N << ": ";
		int64_t cpuT = 0, gpuTRow = 0, gpuTSquare = 0, gpuT = 0;
		LARGE_INTEGER freq = { 0 }, tStartStop[2];
		if (!QueryPerformanceFrequency(&freq)) {
			throw std::runtime_error("Performance counter error");
		}
		for (int i = 0; i < NIter; i++) {

			createRandArr(dataA, numElem);
			createRandArr(dataB, numElem);
			A1->fill(dataA);
			B1->fill(dataB);
			A2->fill(dataA);
			B2->fill(dataB);
			ZeroMemory(tStartStop, sizeof(tStartStop));
			QueryPerformanceCounter(tStartStop);
			C1->mult(A1, B1);
			QueryPerformanceCounter(tStartStop + 1);
			cpuT += tStartStop[1].QuadPart - tStartStop[0].QuadPart;
			ZeroMemory(tStartStop, sizeof(tStartStop));
			QueryPerformanceCounter(tStartStop);
			C2->mult(A2, B2);
			QueryPerformanceCounter(tStartStop + 1);
			gpuT += tStartStop[1].QuadPart - tStartStop[0].QuadPart;
			ZeroMemory(tStartStop, sizeof(tStartStop));
			QueryPerformanceCounter(tStartStop);
			C3->shMultRow(A2, B2);
			QueryPerformanceCounter(tStartStop + 1);
			gpuTRow += tStartStop[1].QuadPart - tStartStop[0].QuadPart;
			ZeroMemory(tStartStop, sizeof(tStartStop));
			QueryPerformanceCounter(tStartStop);
			C4->shMultSquare(A2, B2);
			QueryPerformanceCounter(tStartStop + 1);
			gpuTSquare += tStartStop[1].QuadPart - tStartStop[0].QuadPart;
			std::cout << i << " ";
		}
		std::cout << "\n";
		long double avgcpuT = ((long double)cpuT / (freq.QuadPart)) * 1000.0 / NIter;
		long double avggpuT = ((long double)gpuT / (freq.QuadPart)) * 1000.0 / NIter;
		long double avggpuTRow = ((long double)gpuTRow / (freq.QuadPart)) * 1000.0 / NIter;
		long double avggpuTSquare = ((long double)gpuTSquare / (freq.QuadPart)) * 1000.0 / NIter;
		std::cout << "CPU: " << avgcpuT << " ms.\n";
		std::cout << "GPU non shared: " << avggpuT << " ms.\n";
		std::cout << "GPU shared row: " << avggpuTRow << " ms.\n";
		std::cout << "GPU shared square: " << avggpuTSquare << " ms.\n";
		fprintf(fp, "%i %f %f %f %f\n", N, avgcpuT, avggpuT, avggpuTRow, avggpuTSquare);
		delete A1;
		delete B1;
		delete C1;
		delete A2;
		delete B2;
		delete C2;
		delete C3;
		delete C4;
		changeN *= koefN;
	}
	fclose(fp);
}