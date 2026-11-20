#include "matrix_ops.h"

float **matmul(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols)
{
    // Inner dimensions must match
    if (A_cols != B_rows)
    {
        return NULL;
    }

    // C is A_rows x B_cols, each row allocated separately
    float **C = malloc(A_rows * sizeof(*C));
    for (int i = 0; i < A_rows; i++)
    {
        C[i] = malloc(B_cols * sizeof(**C));
    }

    for (int i = 0; i < A_rows; i++)
    {
        for (int j = 0; j < B_cols; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < A_cols; k++)
            {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    return C;
}

/* ---------------- matmul_thread ---------------- */

// Work assigned to one thread: rows [row_start, row_end) of the output C.
// A and B are shared read-only by all threads; each thread writes only its
// own rows of C, so no locks or other synchronization are needed.
typedef struct
{
    float **A;
    float **B;
    float **C;
    int row_start;
    int row_end;
    int A_cols;
    int B_cols;
} MatmulThreadArgs;

// Thread body: the same i-j-k loop as matmul, restricted to this thread's rows.
// Each C[i][j] is summed over k in the same order as matmul, so the results
// match matmul exactly.
static void *matmul_thread_worker(void *arg)
{
    MatmulThreadArgs *w = (MatmulThreadArgs *)arg;

    for (int i = w->row_start; i < w->row_end; i++)
    {
        for (int j = 0; j < w->B_cols; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < w->A_cols; k++)
            {
                sum += w->A[i][k] * w->B[k][j];
            }
            w->C[i][j] = sum;
        }
    }

    return NULL;
}

// Number of threads to use: MATMUL_DEFAULT_THREADS (the 8 cores Slurm gives
// us), or the MATMUL_THREADS environment variable if set to a positive value
// (used only by the profiling scripts to sweep 1, 2, 4, 8 threads).
static int matmul_num_threads(void)
{
    const char *env = getenv("MATMUL_THREADS");
    if (env != NULL)
    {
        int n = atoi(env);
        if (n > 0)
        {
            return n;
        }
    }
    return MATMUL_DEFAULT_THREADS;
}

float **matmul_thread(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols)
{
    // Inner dimensions must match
    if (A_cols != B_rows)
    {
        return NULL;
    }

    // Allocate C (A_rows x B_cols). To avoid false sharing between threads,
    // every row starts on its own 64-byte cache line and its length is rounded
    // up to a whole number of cache lines. Two different rows therefore never
    // share a cache line, and since threads own whole rows, no cache line is
    // ever written by two threads. posix_memalign memory is released with
    // free(), like the rows of matmul.
    float **C = malloc(A_rows * sizeof(*C));
    size_t row_bytes = (size_t)B_cols * sizeof(float);
    size_t padded_bytes = (row_bytes + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES * CACHE_LINE_BYTES;
    if (padded_bytes == 0)
    {
        padded_bytes = CACHE_LINE_BYTES;
    }
    for (int i = 0; i < A_rows; i++)
    {
        void *row = NULL;
        if (posix_memalign(&row, CACHE_LINE_BYTES, padded_bytes) != 0)
        {
            row = NULL;
        }
        C[i] = (float *)row;
    }

    // Never start more threads than there are output rows
    int num_threads = matmul_num_threads();
    if (num_threads > A_rows)
    {
        num_threads = A_rows;
    }
    if (num_threads < 1)
    {
        return C; // A_rows == 0: nothing to compute
    }

    pthread_t threads[num_threads];
    MatmulThreadArgs args[num_threads];

    // Split the rows of C into num_threads contiguous blocks. Each block gets
    // A_rows / num_threads rows; the first (A_rows % num_threads) blocks get
    // one extra row, so block sizes differ by at most one.
    int base = A_rows / num_threads;
    int extra = A_rows % num_threads;
    int row = 0;
    for (int t = 0; t < num_threads; t++)
    {
        int rows_here = base + (t < extra ? 1 : 0);
        args[t] = (MatmulThreadArgs){A, B, C, row, row + rows_here, A_cols, B_cols};
        row += rows_here;
        pthread_create(&threads[t], NULL, matmul_thread_worker, &args[t]);
    }

    // Wait for every thread to finish its rows before returning C
    for (int t = 0; t < num_threads; t++)
    {
        pthread_join(threads[t], NULL);
    }

    return C;
}
