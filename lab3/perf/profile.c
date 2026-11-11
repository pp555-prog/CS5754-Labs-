// Profiling driver for Lab 3: naive matmul (Lab 1 baseline) vs. matmul_sparse.
//
// Usage: ./perf/profile <matmul|matmul_sparse> <size> <density> <iters>
//   size    : N for (N x N) * (N x N)
//   density : fraction of nonzeros in A, in [0, 1] (sparsity = 1 - density).
//             B is always dense.
//   iters   : number of timed calls
//
// Inputs are generated once with a fixed seed, so both kernels see identical
// matrices for a given (size, density). For matmul_sparse, part 1 (building the
// CSR of A) is setup and is done once, outside the timed loop. Only part 2
// (csr_matmul) is looped, as the handout requires. Run with iters = 0 to
// profile the setup alone and quantify its overhead.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "kernel/kernel.h"

static float rand_float(void)
{
    // Uniform in [-1, 1]
    return 2.0f * (float)rand() / RAND_MAX - 1.0f;
}

// rows x cols matrix (each row malloc'd); each entry is nonzero with probability
// `density`. The value is drawn for every entry, so matrices of different
// densities share the same random stream layout.
static float **rand_matrix(int rows, int cols, double density)
{
    float **m = malloc(rows * sizeof(*m));
    for (int i = 0; i < rows; i++)
    {
        m[i] = malloc(cols * sizeof(**m));
        for (int j = 0; j < cols; j++)
        {
            float v = rand_float();
            double u = (double)rand() / ((double)RAND_MAX + 1.0);
            m[i][j] = (u < density && v != 0.0f) ? v : 0.0f;
        }
    }
    return m;
}

static void free_matrix(float **m, int rows)
{
    for (int i = 0; i < rows; i++)
    {
        free(m[i]);
    }
    free(m);
}

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
    if (argc < 5)
    {
        fprintf(stderr, "Usage: %s <matmul|matmul_sparse> <size> <density> <iters>\n", argv[0]);
        return 1;
    }
    const char *name = argv[1];
    int n = atoi(argv[2]);
    double density = atof(argv[3]);
    long iters = atol(argv[4]);
    int sparse = strcmp(name, "matmul_sparse") == 0;
    if (!sparse && strcmp(name, "matmul") != 0)
    {
        fprintf(stderr, "Unknown kernel: %s\n", name);
        return 1;
    }

    srand(5754); // fixed seed: identical inputs across trials and kernels
    float **A = rand_matrix(n, n, density);
    float **B = rand_matrix(n, n, 1.0);

    float checksum = 0.0f;
    double csr_build_s = 0.0;
    int nnz = 0;
    CSRMatrix *csr = NULL;

    if (sparse)
    {
        // Part 1 (setup, not looped): CSR of A
        double t0 = now_sec();
        csr = csr_from_dense(A, n, n);
        csr_build_s = now_sec() - t0;
        nnz = csr->nnz;
    }
    else
    {
        for (int i = 0; i < n; i++)
            for (int k = 0; k < n; k++)
                nnz += A[i][k] != 0.0f;
    }

    double start = now_sec();
    for (long it = 0; it < iters; it++)
    {
        // Part 2 only for matmul_sparse; the whole kernel for matmul
        float **C = sparse ? csr_matmul(csr, B, n)
                           : matmul(A, B, n, n, n, n);
        checksum += C[n / 2][n / 3];
        free_matrix(C, n);
    }
    double elapsed = now_sec() - start;

    csr_free(csr);
    free_matrix(A, n);
    free_matrix(B, n);

    printf("RESULT kernel=%s size=%d density=%g nnz=%d iters=%ld total_s=%.6f per_call_us=%.3f csr_build_us=%.3f checksum=%f\n",
           name, n, density, nnz, iters, elapsed, iters > 0 ? elapsed / iters * 1e6 : 0.0,
           csr_build_s * 1e6, checksum);
    return 0;
}
