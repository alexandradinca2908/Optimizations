/*
 * Tema 2 ASC
 * 2025 Spring
 */
#include "utils.h"

/*
 * Add your optimized implementation here
 */
double* my_solver(int N, double *A, double *B, double *x) {
    register double suma = 0.0;

    //  C = B * At
    //  For sequencial access, we don't transpose A
    double *C = malloc(N * N * sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            suma = 0.0;
            for (int k = 0; k < N; k++) {
                suma += B[i * N + k] * A[j * N + k];
            }
            C[i * N + j] = suma;
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
            suma = 0.0;
            for (int k = 0; k < N; k++) {
                suma += Ct[i * N + k] * A[k * N + j];
            }
            D[i * N + j] = suma;
        }
    }

    double *y = malloc(N * sizeof(double));
    for (int iter = 0; iter < N; iter++) {
        //  y = Ct * x
        for (int i = 0; i < N; i++) {
            suma = 0.0;
            for (int j = 0; j < N; j++) {
                suma += Ct[i * N + j] * x[j];
            }
            y[i] = suma;
        }

        //  x = C * y
        for (int i = 0; i < N; i++) {
            suma = 0.0;
            for (int j = 0; j < N; j++) {
                suma += C[i * N + j] * y[j];
            }
            x[i] = suma;
        }
    }

    //  y = D * x
    for (int i = 0; i < N; i++) {
        suma = 0.0;
        for (int j = 0; j < N; j++) {
            suma += D[i * N + j] * x[j];
        }
        y[i] = suma;
    }

    //  Free memory
    free(C);
    free(D);
    free(Ct);

    return y;
}