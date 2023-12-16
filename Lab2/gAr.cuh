#pragma once
#include"cAr.h"

class gAr {
    int dimNumb = 0;
    int elemNumb = 0;
    int* dimVal;
    double* data;
    int currElem = 0;
    //   bool isfull = false;

public:
    gAr(int ndim, const int* shape, const double* postData = nullptr);
    void fill(const double* moreValue);

    gAr operator+=(const gAr* B);
    gAr operator+(const gAr* B);
    void plus(const gAr* B);
    double error(const gAr* B);
    double error(const cAr* B);
    void mult(const gAr* A, const gAr* B);
    void shMultSquare(const gAr* A, const gAr* B);
    void shMultRow(const gAr* A, const gAr* B);
    void Print(int dimention = 0);
    void PrintReccur(int dimention, double* data);

    ~gAr();
};