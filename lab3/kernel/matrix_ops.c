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

/**** Part 1: build the CSR representation of a dense matrix ****/
CSRMatrix *csr_from_dense(float **A, int rows, int cols)
{
    CSRMatrix *csr = malloc(sizeof(*csr));
    csr->rows = rows;
    csr->cols = cols;

    // Pass 1: count the nonzeros of each row. row_ptr[i + 1] is the running
    // (cumulative) count after row i, so row_ptr[0] = 0 and row_ptr[rows] = nnz.
    csr->row_ptr = malloc((rows + 1) * sizeof(*csr->row_ptr));
    csr->row_ptr[0] = 0;
    for (int i = 0; i < rows; i++)
    {
        int count = 0;
        for (int k = 0; k < cols; k++)
        {
            if (A[i][k] != 0.0f)
            {
                count++;
            }
        }
        csr->row_ptr[i + 1] = csr->row_ptr[i] + count;
    }
    csr->nnz = csr->row_ptr[rows];

    // Pass 2: copy each nonzero and its column index, row by row with columns
    // in ascending order. The (nnz + 1) keeps malloc's size nonzero for an
    // all-zero matrix.
    csr->values = malloc((csr->nnz + 1) * sizeof(*csr->values));
    csr->col_idx = malloc((csr->nnz + 1) * sizeof(*csr->col_idx));
    int p = 0;
    for (int i = 0; i < rows; i++)
    {
        for (int k = 0; k < cols; k++)
        {
            if (A[i][k] != 0.0f)
            {
                csr->values[p] = A[i][k];
                csr->col_idx[p] = k;
                p++;
            }
        }
    }

    return csr;
}

/**** Part 2: multiply a CSR matrix A (rows x cols) by a dense B (cols x B_cols) ****/
float **csr_matmul(const CSRMatrix *A, float **B, int B_cols)
{
    // C is A->rows x B_cols, each row allocated separately and zeroed, because
    // every nonzero of A adds a scaled row of B into C
    float **C = malloc(A->rows * sizeof(*C));
    for (int i = 0; i < A->rows; i++)
    {
        C[i] = calloc(B_cols, sizeof(**C));
    }

    // Row i of C is the sum over the nonzeros A[i][k] of A[i][k] * (row k of B).
    // Only the stored nonzeros are visited, so the work is nnz * B_cols
    // multiply-adds instead of rows * cols * B_cols: every multiplication by
    // a zero of A is skipped.
    for (int i = 0; i < A->rows; i++)
    {
        float *C_row = C[i];
        for (int p = A->row_ptr[i]; p < A->row_ptr[i + 1]; p++)
        {
            // Nonzero A[i][k]: kept in a register while the inner loop walks
            // row k of B and row i of C with unit stride (sequential accesses).
            float a = A->values[p];
            float *B_row = B[A->col_idx[p]];
            for (int j = 0; j < B_cols; j++)
            {
                C_row[j] += a * B_row[j];
            }
        }
        // Nonzeros are visited in ascending k, so each C[i][j] adds the same
        // terms in the same order as matmul's sum (minus the zero terms), and
        // the results match matmul exactly.
    }

    return C;
}

void csr_free(CSRMatrix *csr)
{
    if (csr == NULL)
    {
        return;
    }
    free(csr->values);
    free(csr->col_idx);
    free(csr->row_ptr);
    free(csr);
}

// Matmul that skips the zeros of A by working on its CSR form
float **matmul_sparse(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols)
{
    // Inner dimensions must match
    if (A_cols != B_rows)
    {
        return NULL;
    }

    /**** 1. Create CSR format of input matrix ****/
    CSRMatrix *csr = csr_from_dense(A, A_rows, A_cols);

    /**** 2. Perform matrix multiplication on CSR format of input matrix ****/
    // When profiling, only part 2 is looped (perf/profile.c calls csr_matmul
    // directly on a CSR built once).
    float **result = csr_matmul(csr, B, B_cols);

    csr_free(csr);
    return result;
}
