#pragma once

class cAr {
    int dimNumb = 0;
    int elemNumb = 0;
    int* dimVal;
    double* data;
    int currElem = 0;
 //   bool isfull = false;

public:
    cAr(int ndim, const int* shape, const double* postData = nullptr);
    void fill(const double* moreValue);

    cAr operator+=(const cAr* B);
    cAr operator+(const cAr* B);
    void plus(const cAr* B);
    void Print(int dimention = 0);

    ~cAr();
};