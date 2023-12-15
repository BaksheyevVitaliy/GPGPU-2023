#include <iostream>

#include"cAr.h"
#include"gAr.cuh"




int main(void) {
	int secondArDim[4] = { 1, 3, 2, 2 };
	double secondArDataA[12] = { 0.5, 8 ,3.4, 1.9, 4.7, 2.1, 3.7, 3.1, 5.8, 1.3, 5.4, 1.2};
	double secondArDataB[12] = { -0.5, 3.2, 6.7, 3.9, 2.5, 3.8, 9.1, 2.4, 6.8, 2.4, 2.1, 2.7 };
	cAr* A2 = new cAr(4, secondArDim, secondArDataA);
	cAr* B2 = new cAr(4, secondArDim, secondArDataB);
	std::cout << "CPU:\n";
	std::cout << "A:\n";
	A2->Print();
	std::cout << "B2:\n";
	B2->Print();
	A2->plus(B2);
	std::cout << "A2+B2:\n";
	A2->Print();
	gAr* A = new gAr(4, secondArDim, secondArDataA);
	gAr* B = new gAr(4, secondArDim, secondArDataB);
	std::cout << "GPU:\n";
	std::cout << "A:\n";
	A->Print();
	std::cout << "B2:\n";
	B->Print();
	A->plus(B);
	std::cout << "A2+B2:\n";
	A->Print();
	A->fill(secondArDataB);
	A->Print();
	delete A;
	delete B;
	delete A2;
	delete B2;
}