#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <cblas.h>

// Threads used by matmul_thread (the Slurm job gets 8 cores). The
// MATMUL_THREADS environment variable overrides it, for profiling sweeps.
#define MATMUL_DEFAULT_THREADS 8

// Output rows of matmul_thread are aligned and padded to this size so that no
// cache line is shared between two threads (avoids false sharing).
#define CACHE_LINE_BYTES 64

float **matmul(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols);
float **matmul_thread(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols);

#endif /* MATRIX_OPS_H */
