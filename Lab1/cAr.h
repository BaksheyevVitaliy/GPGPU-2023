#pragma once
class cAr {
    int dimNumb = 0;
    int elemNumb = 0;
    int* dimVal;
    double* data;
    double currElem = 0;

public:
    cAr(int ndim, const int* shape, const double* postData = nullptr);
    void fill(const double value);
    void fill(const double* moreValue);
    void copy_data(const cAr* source);
    void print() const;

    cAr operator[](int idx);
    cAr operator=(const cAr* ndarray);
    cAr operator+=(const cAr* ndarray);
    cAr operator+(const cAr* ndarray);
   
    size_t get_ndim() const { return dimNumb; }
    size_t get_elems_num() const { return elemNumb; }

    ~cAr();
};