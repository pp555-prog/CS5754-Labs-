#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <stdio.h>
#include <stdlib.h>

float **matmul(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols);
float **matmul_sparse(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols);

// Compressed Sparse Row (CSR) matrix. Only the nonzero entries are stored:
//   values[p]  : the p-th nonzero, in row-major order (row by row, columns ascending)
//   col_idx[p] : the column of values[p]
//   row_ptr[i] : index in values/col_idx of the first nonzero of row i;
//                row i's nonzeros are p = row_ptr[i] .. row_ptr[i+1]-1.
//                row_ptr has rows + 1 entries, row_ptr[0] = 0, row_ptr[rows] = nnz.
typedef struct
{
    int rows;
    int cols;
    int nnz;
    float *values;
    int *row_ptr;
    int *col_idx;
} CSRMatrix;

// The two parts of matmul_sparse, exposed separately so that a profiling driver
// can build the CSR once (setup) and loop only over the multiplication.
CSRMatrix *csr_from_dense(float **A, int rows, int cols);   // part 1
float **csr_matmul(const CSRMatrix *A, float **B, int B_cols); // part 2
void csr_free(CSRMatrix *csr);

#endif /* MATRIX_OPS_H */
