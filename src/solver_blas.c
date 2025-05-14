/*
 * Tema 2 ASC
 * 2024 Spring
 */
#include "utils.h"
#include <cblas.h>

/* 
 * Add your BLAS implementation here
 */
double* my_solver(int N, double *A, double *B, double *x) {
    //  C = B * At
    double *C = (double *)malloc(N * N * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                N, N, N, 1.0, B, N, A, N, 0.0, C, N);

    //  D = Ct * A
    double *D = (double *)malloc(N * N * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                N, N, N, 1.0, C, N, A, N, 0.0, D, N);

    double *y = (double *)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        //  y = Ct * x
        cblas_dgemv(CblasRowMajor, CblasTrans,
                    N, N, 1.0, C, N, x, 1, 0.0, y, 1);

        //  x = C * y
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    N, N, 1.0, C, N, y, 1, 0.0, x, 1);
    }

    //  y = D * x
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                N, N, 1.0, D, N, x, 1, 0.0, y, 1);

    free(C);
    free(D);

	return y;
}