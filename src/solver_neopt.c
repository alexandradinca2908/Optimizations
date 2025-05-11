/*
 * Tema 2 ASC
 * 2025 Spring
 */
#include "utils.h"

/*
 * Add your unoptimized implementation here
 */
double* my_solver(int N, double *A, double *B, double *x) {
    //  Transpose A
    double *At = malloc(N * N * sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            At[i * N + j] = A[j * N + i];
        }
    }

    //  C = B * At
    double *C = malloc(N * N * sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0;
            for (int k = 0; k < N; k++) {
                C[i * N + j] += B[i * N + k] * At[k * N + j];
            }
        }
    }

    //  Transpose C
    double *Ct = malloc(N * N * sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            Ct[i * N + j] = C[j * N + i];
        }
    }

    //  D = Ct * A
    double *D = malloc(N * N * sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            D[i * N + j] = 0;
            for (int k = 0; k < N; k++) {
                D[i * N + j] += Ct[i * N + k] * A[k * N + j];
            }
        }
    }

    double *y = malloc(N * sizeof(double));
    for (int iter = 0; iter < N; iter++) {
        //  y = Ct * x
        for (int i = 0; i < N; i++) {
            y[i] = 0;
            for (int j = 0; j < N; j++) {
                y[i] += Ct[i * N + j] * x[j];
            }
        }

        //  x = C * y
        for (int i = 0; i < N; i++) {
            x[i] = 0;
            for (int j = 0; j < N; j++) {
                x[i] += C[i * N + j] * y[j];
            }
        }
    }

    //  y = D * x
    for (int i = 0; i < N; i++) {
        y[i] = 0;
        for (int j = 0; j < N; j++) {
            y[i] += D[i * N + j] * x[j];
        }
    }

    //  Free memory
    free(C);
    free(D);
    free(At);
    free(Ct);

    return y;
}