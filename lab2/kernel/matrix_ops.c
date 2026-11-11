#include "matrix_ops.h"

// Tile sizes for matmul_blocking (override with -DBLOCK_SIZE_I=... etc.).
// 32 x 256 x 32 was the fastest of the configurations swept on the ece5755
// node (Sapphire Rapids, 48 KB L1d, 2 MB L2) for 512 and 1536 square inputs:
// a 32 x 256 tile of B (32 KB) fits in L1d alongside the active rows of A and
// C, and the 256-wide j loop gives long unit-stride runs over B and C.
#ifndef BLOCK_SIZE_I
#define BLOCK_SIZE_I 32
#endif

#ifndef BLOCK_SIZE_J
#define BLOCK_SIZE_J 256
#endif

#ifndef BLOCK_SIZE_K
#define BLOCK_SIZE_K 32
#endif


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

// Matmul with blocking optimization
float **matmul_blocking(float **A, float **B, int A_rows, int A_cols, int B_rows, int B_cols)
{
    // Inner dimensions must match
    if (A_cols != B_rows)
    {
        return NULL;
    }

    // C is A_rows x B_cols, each row allocated separately and zeroed,
    // because every (kk) tile adds a partial sum into C
    float **C = malloc(A_rows * sizeof(*C));
    for (int i = 0; i < A_rows; i++)
    {
        C[i] = calloc(B_cols, sizeof(**C));
    }

    // Outer loops walk over tiles: a BLOCK_SIZE_I x BLOCK_SIZE_K tile of A
    // times a BLOCK_SIZE_K x BLOCK_SIZE_J tile of B accumulates into a
    // BLOCK_SIZE_I x BLOCK_SIZE_J tile of C. The tiles are sized so the working
    // set stays in cache while it is reused, instead of streaming all of B
    // (column by column) for every row of A as the naive i-j-k matmul does.
    for (int ii = 0; ii < A_rows; ii += BLOCK_SIZE_I)
    {
        // Edge tiles are clipped when a dimension is not a multiple of the tile size
        int i_end = (ii + BLOCK_SIZE_I < A_rows) ? ii + BLOCK_SIZE_I : A_rows;

        for (int kk = 0; kk < A_cols; kk += BLOCK_SIZE_K)
        {
            int k_end = (kk + BLOCK_SIZE_K < A_cols) ? kk + BLOCK_SIZE_K : A_cols;

            for (int jj = 0; jj < B_cols; jj += BLOCK_SIZE_J)
            {
                int j_end = (jj + BLOCK_SIZE_J < B_cols) ? jj + BLOCK_SIZE_J : B_cols;

                // Inside a tile use i-k-j order: A[i][k] is loaded once and kept
                // in a register, and the innermost loop walks rows of B and C
                // with unit stride (sequential, cache-line friendly accesses).
                for (int i = ii; i < i_end; i++)
                {
                    float *C_row = C[i];
                    for (int k = kk; k < k_end; k++)
                    {
                        float a = A[i][k];
                        float *B_row = B[k];
                        for (int j = jj; j < j_end; j++)
                        {
                            C_row[j] += a * B_row[j];
                        }
                    }
                }
            }
        }
    }

    return C;
}

// Matmul with quantization
float **matmul_quant(unsigned int **A, int **B, int A_rows, int A_cols, int B_rows, int B_cols)
{
    // TODO: Implement matrix multiplication with quantized matrix A and unquantized matrix B.
    //       The input matrix A stores data in a 4-byte granularity, i.e., A[i][k] is a 4-byte unsigned integer.
    //       However, each element of A takes 8-bit (1-byte) storage space, meadning that A[i][k] contains multiple elements.
    //       The input matrix B is unquantized, and each element of B is a 4-byte integer.
    //       The returned output matrix should be "float". You can achieve this by performing dot product in integer, and use type casting to convert the output to float.

    if ( ((QUANT_PRECISION % 2) != 0) || (QUANT_PRECISION > 16) )
    {
        printf("Incompatible precision assigned to the matrix.\n");
        return NULL;
    }

    if ( (A_cols * (sizeof(unsigned int)*8) / QUANT_PRECISION) != B_rows )
    {
        printf("Matrix dimensions incompatible for multiplication.\n");
        return NULL;
    }

    // Number of quantized elements packed in one word (4 for 8-bit), and the
    // mask that keeps the low QUANT_PRECISION bits of a shifted word
    int elems_per_word = (sizeof(unsigned int) * 8) / QUANT_PRECISION;
    unsigned int mask = (1u << QUANT_PRECISION) - 1;

    float **C = malloc(A_rows * sizeof(*C));
    for (int i = 0; i < A_rows; i++)
    {
        C[i] = malloc(B_cols * sizeof(**C));
    }

    for (int i = 0; i < A_rows; i++)
    {
        for (int j = 0; j < B_cols; j++)
        {
            // Integer dot product, converted to float at the end
            long long sum = 0;
            for (int k = 0; k < A_cols; k++)
            {
                unsigned int word = A[i][k];
                // Elements are packed most-significant first: 0x01020304 holds
                // [1, 2, 3, 4], so element e sits (elems_per_word - 1 - e) slots
                // above the least-significant bits. It multiplies row
                // k * elems_per_word + e of B.
                for (int e = 0; e < elems_per_word; e++)
                {
                    int shift = (elems_per_word - 1 - e) * QUANT_PRECISION;
                    unsigned int a = (word >> shift) & mask;
                    sum += (long long)a * B[k * elems_per_word + e][j];
                }
            }
            C[i][j] = (float)sum;
        }
    }

    return C;
}