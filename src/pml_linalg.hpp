#ifndef MATLIB_PML_LINALG_H
#define MATLIB_PML_LINALG_H

#include "pml.hpp"

extern "C" {
// LU decomoposition of a general matrix
void dgetrf_(int*, int*, double*, int*, int*, int*);

// generate inverse of a matrix given its LU decomposition
void dgetri_(int*, double*, int*, int*, double*, int*, int* );
}

namespace pml{

  Matrix Inv(const Matrix &matrix) {
    Matrix result(matrix);
    int N = matrix.num_rows();
    int *IPIV = new int[N+1];
    int LWORK = N*N;
    double *WORK = new double[LWORK];
    int INFO;
    dgetrf_(&N,&N,result.data(),&N,IPIV,&INFO);
    dgetri_(&N,result.data(),&N,IPIV,WORK,&LWORK,&INFO);
    delete IPIV;
    delete WORK;
    return result;
  }

} // pml

#endif