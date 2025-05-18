Name: Dincă Alexandra Cristina
Grupa: 334CD

Matrix Calculus Optimizations: Memory access optimizations using 2 methods, BLAS and manual optimizations, which can be compared to the classic way of computing matrix multiplication.

1. Cachegrind Analysis
    Comparing all 3, the BLAS implementation executes many more instructions and performs many more data accesses than the manual implementations. This is because it uses advanced algorithms and specialized code. As a result, overall cache miss rate is lower than the other 2:
        D1 miss rate: 2.5% for BLAS vs. 4.9% for manual implementations
        LLd miss rate: 0.8% for BLAS vs. 4.1% for manual implementations
        Branch misprediction rate: 4.7% for BLAS vs 12.3% for manual implementations

    Comparison between neopt and opt_m:
        Instruction References (I refs):
            Both implementations show exactly the same number of instruction references: 236,227. This is because the optimizations in opt_m didn't reduce the total number of instructions that were executed, but improved how they were executed.

        Data References (D refs):
            Both implementations have identical data references: 67,819, because they are accessing the same amount of data.

        Cache Misses:
            I1 misses: Instruction cache misses are very simillar (1,316 vs 1,319)
            D1 misses: Data cache misses are slightly better in the optimized version (3,294 vs 3,300)
            LL misses: Last-level cache misses are a bit better in the optimized version (4,023 vs 4,031)

        Branch Prediction:
            Both implementations have almost identical branch statistics: 46,637 branches, with one less misspredict in the opt_m version (5,717)

        Overall, the optimized version has fewer cache misses.

        Impact of manual optimizations: the manual optimizations had minimal impact on cache performance. The key differences between the algorithms are register usage and avoiding transposition (to increase linearity and avoid further operations). These optimizations help with cache misses, but in small amounts (probably due to the compiler already optimizing the code itself).

2. Comparing Performances:
    neopt: this is the basic version of matrix multiplication. It has many flaws due to slow and disadvantageous memory access.

    BLAS: this solution is the most optimal one. It uses sophisticated functions that greatly reduce cache misses and improve memory access, therefore providing major speed-ups.

    opt_m: this is the optimized version of neopt. Though not as good as BLAS, it has a few enhancements that prove to make a great difference, especially for larger N values. Firstly, it uses a register variable to compute all the sums. This way, memory access is less frequent and operations become faster. Secondly, getting rid of At helps because: we use less storage, we do less operations, we reduce memory access and we linearize column access for C * At. Lastly, I'd like to mention that I tried changing loop orders (ijk -> ikj), but having to get rid of the sum register just increased the overall time.

3. Graph analysis:
    The bar chart shows how well BLAS performs. Compared to the other algorithms, BLAS increases in time way slower than the other 2. Moreover, it can be seen how much of a difference opt_m makes compared to neopt. Due to logarithmic scale, the bars are a bit uneven; but for the largest test, the manual optimizations reduce execution time by HALF.

4. LLM prompts:
    All of these prompts were discussed with ChatGPT. I have the free version, which allows me to use the premium chat bot for a few messages.

    1) how to use blas functions in c -> I wanted a tutorial over function usage to understand BLAS better (I tried to look it up on Google first but there were little to no explanations)

        1. Install ATLAS (if not already installed)
    On Ubuntu/Debian:

    bash
    Copy
    Edit
    sudo apt install libatlas-base-dev
    This provides both:

    The BLAS library (Fortran-style and C interface)

    The C header: cblas.h

    ✅ 2. Include cblas.h in your C code
    c
    Copy
    Edit
    #include <cblas.h>
    ✅ 3. Example: Use cblas_dgemm with ATLAS
    Here's a simple matrix multiplication example (2×2 matrices):

    c
    Copy
    Edit
    #include <stdio.h>
    #include <cblas.h>

    int main() {
        double A[4] = {1, 2, 3, 4}; // 2x2 matrix
        double B[4] = {5, 6, 7, 8}; // 2x2 matrix
        double C[4] = {0};          // Result matrix

        int m = 2, n = 2, k = 2;

        // C = A * B
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, 1.0, A, k, B, n, 0.0, C, n);

        printf("Result:\n");
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                printf("%f ", C[i*n + j]);
            }
            printf("\n");
        }

        return 0;
    }
    ✅ 4. Compile with ATLAS
    Use -lcblas -latlas:

    bash
    Copy
    Edit
    gcc myblas.c -o myblas -lcblas -latlas
    If needed, add math lib:

    bash
    Copy
    Edit
    gcc myblas.c -o myblas -lcblas -latlas -lm
    🔁 Optional: Use Level 1 and 2 BLAS
    For example, vector addition with cblas_daxpy:

    c
    Copy
    Edit
    int n = 3;
    double alpha = 2.0;
    double X[3] = {1, 2, 3};
    double Y[3] = {4, 5, 6};  // Y = alpha * X + Y

    cblas_daxpy(n, alpha, X, 1, Y, 1);
    After the call, Y becomes {6, 9, 12}.

    Do you want a quick cheat sheet with the most common ATLAS BLAS calls?

    2) "Do you want a quick cheat sheet with the most common ATLAS BLAS calls?" You said: yes -> generated concise information regarding BLAS operations

    Here's a concise BLAS (ATLAS) cheat sheet for C, using the CBLAS interface (<cblas.h>), with typical use cases and signatures:

    🔹 Level 1 BLAS: Vector Operations
    Function	Operation	Signature	Example
    cblas_daxpy	Y ← α·X + Y	void cblas_daxpy(int n, double alpha, const double *X, int incX, double *Y, int incY);	Add scaled vector
    cblas_ddot	Dot product	double cblas_ddot(int n, const double *X, int incX, const double *Y, int incY);	double result = cblas_ddot(n, X, 1, Y, 1);
    cblas_dcopy	Copy X to Y	void cblas_dcopy(int n, const double *X, int incX, double *Y, int incY);	
    cblas_dscal	X ← α·X (scaling)	void cblas_dscal(int n, double alpha, double *X, int incX);	

    🔹 Level 2 BLAS: Matrix-Vector Operations
    Function	Operation	Signature	Example
    cblas_dgemv	Y ← α·A·X + β·Y	void cblas_dgemv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA, int M, int N, double alpha, const double *A, int lda, const double *X, int incX, double beta, double *Y, int incY);	Multiply matrix by vector

    🔹 Level 3 BLAS: Matrix-Matrix Operations
    Function	Operation	Signature	Example
    cblas_dgemm	C ← α·A·B + β·C	void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, int M, int N, int K, double alpha, const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc);	Matrix multiplication

    🛠️ Example Enums
    c
    Copy
    Edit
    CblasRowMajor        // Row-major ordering (C style)
    CblasColMajor        // Column-major ordering (Fortran style)

    CblasNoTrans         // Use A as-is
    CblasTrans           // Use Aᵀ (transpose)
    CblasConjTrans       // Use conjugate transpose
    ✅ Tips
    Always link with: -lcblas -latlas

    Double precision = d prefix (dgemm, daxpy, etc.)

    Single precision = s prefix (sgemm, saxpy, etc.)

    Use incX/incY = 1 for normal arrays

    Would you like this as a printable PDF or markdown cheat sheet too?

    3) what s K, lda, ldb, ldc? -> needed further explanations

    Great question! These parameters are crucial for understanding how cblas_dgemm works. Here's a breakdown using matrix dimensions and layout:

    ✅ General Form of DGEMM
    The function performs the operation:

    ini
    Copy
    Edit
    C = α * op(A) * op(B) + β * C
    Where:

    op(A) is either A or its transpose (Aᵀ)

    op(B) is either B or Bᵀ

    α and β are scalars

    A, B, and C are matrices stored in memory

    🧮 Matrix Dimensions and Parameters
    Let’s assume row-major order (CblasRowMajor) and no transpose (CblasNoTrans):

    Param	Meaning	Typical Value
    M	Rows of A and C	#rows of A
    N	Columns of B and C	#cols of B
    K	Columns of A = Rows of B	shared dimension

    🧱 Leading Dimensions: lda, ldb, ldc
    These tell BLAS how many elements are between rows of a matrix in memory.

    In row-major:

    lda = K → because in memory, each row of A has K elements

    ldb = N → each row of B has N elements

    ldc = N → each row of C has N elements

    So if your matrices are defined like this in C:

    c
    Copy
    Edit
    double A[M * K];
    double B[K * N];
    double C[M * N];
    Then you should use:

    c
    Copy
    Edit
    lda = K;
    ldb = N;
    ldc = N;
    In column-major layout (Fortran style), lda would be the number of rows in A instead.

    📌 Summary Mapping
    For cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, ...):

    Matrix	Dimensions	Stored as 1D array	Leading dimension
    A	M × K	row-major → A[i*K + j]	lda = K
    B	K × N	row-major → B[i*N + j]	ldb = N
    C	M × N	row-major → C[i*N + j]	ldc = N

    Would you like a concrete example with values plugged in for A, B, C?

    4) do this C = B * At -> asked for an example

    #include <stdio.h>
    #include <stdlib.h>
    #include <cblas.h>

    int main() {
        int M = 4; // rows in A
        int K = 3; // cols in A

        // A is MxK
        double A[12] = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
            10, 11, 12
        };

        // B is 1xK
        double B[3] = {1, 0, 1};  // Think of this as a row vector

        // C = B * Aᵀ => 1xM
        double *C = (double *)malloc(M * sizeof(double));

        // We use cblas_dgemv to compute: C = B * Aᵀ
        cblas_dgemv(
            CblasRowMajor,      // Row-major layout
            CblasTrans,         // Transpose A, so it becomes KxM
            M,                  // Number of rows in A
            K,                  // Number of cols in A
            1.0,                // alpha
            A,                  // A matrix
            K,                  // lda = number of cols in A
            B,                  // X = vector (1xK)
            1,                  // incX
            0.0,                // beta
            C,                  // Y = result vector (1xM)
            1                   // incY
        );

        printf("C = B * Aᵀ = [ ");
        for (int i = 0; i < M; i++) {
            printf("%.2f ", C[i]);
        }
        printf("]\n");

        free(C);
        return 0;
    }

    -- afterwards I was able to complete the rest of the task by myself

    5) now solve this in the classic, unoptimized way:
        C = B * At
        D = Ct * A
        i = 0
        while (i < N) {
            y = Ct * x
            x = C * y
            i += 1
        }
        y = D * x
    -> I generated the entire unoptimized code, then cleaned it up. It had unnecessary temporary variables and I changed the allocations to malloc instead of calloc. Logicwise, the algorithm was fine.

    #include "utils.h"
    #include <stdlib.h>
    #include <string.h>

    double* my_solver(int N, double *A, double *B, double *x) {
        // Allocate memory
        double *C = calloc(N * N, sizeof(double));
        double *D = calloc(N * N, sizeof(double));
        double *At = calloc(N * N, sizeof(double));
        double *Ct = calloc(N * N, sizeof(double));
        double *y = calloc(N, sizeof(double));
        double *tmp = calloc(N, sizeof(double)); // temporary vector for swapping

        // Transpose A -> At
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                At[i * N + j] = A[j * N + i];

        // C = B * At
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                for (int k = 0; k < N; k++)
                    C[i * N + j] += B[i * N + k] * At[k * N + j];

        // Transpose C -> Ct
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                Ct[i * N + j] = C[j * N + i];

        // D = Ct * A
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                for (int k = 0; k < N; k++)
                    D[i * N + j] += Ct[i * N + k] * A[k * N + j];

        // Iterative loop
        for (int iter = 0; iter < N; iter++) {
            // y = Ct * x
            for (int i = 0; i < N; i++) {
                y[i] = 0;
                for (int j = 0; j < N; j++)
                    y[i] += Ct[i * N + j] * x[j];
            }

            // x = C * y
            for (int i = 0; i < N; i++) {
                tmp[i] = 0;
                for (int j = 0; j < N; j++)
                    tmp[i] += C[i * N + j] * y[j];
            }

            memcpy(x, tmp, N * sizeof(double));
        }

        // y = D * x
        for (int i = 0; i < N; i++) {
            y[i] = 0;
            for (int j = 0; j < N; j++)
                y[i] += D[i * N + j] * x[j];
        }

        // Free temp arrays
        free(C);
        free(D);
        free(At);
        free(Ct);
        free(tmp);

        // Return the result
        return y;
    }

    -- one prompt was enough to solve this task

    -- for opt_m, I didn't use ChatGPT. I used the information given in lab9 only (https://ocw.cs.pub.ro/courses/asc/laboratoare/09). Out of curiosity, I asked the bot if it has any improvements for neopt, but the results were overly complicated and made the execution time worse :D.

5. while (N) loop explanation:
    The loop can be translated to x = (C * Ct) ^ N * x. The goal is to apply the transformations defined by the matrices C and Ct to the vector x and repeating this N times ensures that the algorithm stabilizes the transformed vector. Also, by increasing the N value, we can see the efficiency of all 3 methods, because the loop increases computational difficulty.
