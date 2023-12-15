#pragma once

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
    void Print(int dimention = 0);
    void PrintReccur(int dimention, double* data);

    ~gAr();
};