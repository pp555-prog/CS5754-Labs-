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
