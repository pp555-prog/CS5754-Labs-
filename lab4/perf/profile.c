// Profiling driver for Lab 4: naive matmul (Lab 1 baseline) vs. matmul_thread.
//
// Usage: [MATMUL_THREADS=<t>] ./perf/profile <matmul|matmul_thread> <size> <iters>
//   size    : N for (N x N) * (N x N)
//   iters   : number of timed calls
// matmul_thread uses 8 threads unless MATMUL_THREADS overrides it.
//
// Inputs are generated once with a fixed seed, so both kernels see identical
// matrices for a given size, and the checksum must match between them. Only
// the kernel call is looped. Run with iters = 0 to profile the setup alone and
// quantify its overhead.
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

// rows x cols random matrix (each row malloc'd)
static float **rand_matrix(int rows, int cols)
{
    float **m = malloc(rows * sizeof(*m));
    for (int i = 0; i < rows; i++)
    {
        m[i] = malloc(cols * sizeof(**m));
        for (int j = 0; j < cols; j++)
        {
            m[i][j] = rand_float();
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
    if (argc < 4)
    {
        fprintf(stderr, "Usage: %s <matmul|matmul_thread> <size> <iters>\n", argv[0]);
        return 1;
    }
    const char *name = argv[1];
    int n = atoi(argv[2]);
    long iters = atol(argv[3]);
    int threaded = strcmp(name, "matmul_thread") == 0;
    if (!threaded && strcmp(name, "matmul") != 0)
    {
        fprintf(stderr, "Unknown kernel: %s\n", name);
        return 1;
    }
    const char *env = getenv("MATMUL_THREADS");
    int threads = threaded ? (env ? atoi(env) : MATMUL_DEFAULT_THREADS) : 1;

    srand(5754); // fixed seed: identical inputs across trials and kernels
    float **A = rand_matrix(n, n);
    float **B = rand_matrix(n, n);

    float checksum = 0.0f;
    double start = now_sec();
    for (long it = 0; it < iters; it++)
    {
        float **C = threaded ? matmul_thread(A, B, n, n, n, n)
                             : matmul(A, B, n, n, n, n);
        checksum += C[n / 2][n / 3];
        free_matrix(C, n);
    }
    double elapsed = now_sec() - start;

    free_matrix(A, n);
    free_matrix(B, n);

    printf("RESULT kernel=%s size=%d threads=%d iters=%ld total_s=%.6f per_call_us=%.3f checksum=%f\n",
           name, n, threads, iters, elapsed, iters > 0 ? elapsed / iters * 1e6 : 0.0, checksum);
    return 0;
}
