#include "unity/unity.h"
#include "../kernel/kernel.h"
#include "test_matrix_ops.h"


#define EPSILON 0.000001f

void assert_float_array_equal_matmul(float **expected, float **actual, int rows, int cols)
{
    TEST_ASSERT_NOT_NULL_MESSAGE(actual, "Function returned NULL (not implemented?)");
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            UNITY_TEST_ASSERT_FLOAT_WITHIN(EPSILON, expected[i][j], actual[i][j], __LINE__, "Arrays Not Equal!");
        }
    }
}

void test_matmul_square_matrices(void)
{
    // Setup
    float **A = (float **)malloc(2 * sizeof(float *));
    for (int i = 0; i < 2; i++)
    {
        A[i] = (float *)malloc(2 * sizeof(float));
    }
    A[0][0] = 1.0f;
    A[0][1] = 2.0f;
    A[1][0] = 3.0f;
    A[1][1] = 4.0f;

    float **B = (float **)malloc(2 * sizeof(float *));
    for (int i = 0; i < 2; i++)
    {
        B[i] = (float *)malloc(2 * sizeof(float));
    }
    B[0][0] = 2.0f;
    B[0][1] = 0.0f;
    B[1][0] = 1.0f;
    B[1][1] = 2.0f;

    float **expected = (float **)malloc(2 * sizeof(float *));
    for (int i = 0; i < 2; i++)
    {
        expected[i] = (float *)malloc(2 * sizeof(float));
    }
    expected[0][0] = 4.0f;
    expected[0][1] = 4.0f;
    expected[1][0] = 10.0f;
    expected[1][1] = 8.0f;

    // Run function under test
    float **C = matmul(A, B, 2, 2, 2, 2);

    // Check expectations
    assert_float_array_equal_matmul(expected, C, 2, 2);

    // Cleanup
    for (int i = 0; i < 2; i++)
    {
        free(A[i]);
        free(B[i]);
        free(C[i]);
        free(expected[i]);
    }
    free(A);
    free(B);
    free(C);
    free(expected);
}

void test_matmul_incompatible_dimensions(void)
{
    // Setup
    float **A = (float **)malloc(2 * sizeof(float *));
    for (int i = 0; i < 2; i++)
    {
        A[i] = (float *)malloc(3 * sizeof(float));
    }

    float **B = (float **)malloc(2 * sizeof(float *));
    for (int i = 0; i < 2; i++)
    {
        B[i] = (float *)malloc(2 * sizeof(float));
    }

    // Run function under test
    float **C = matmul(A, B, 2, 3, 2, 2);

    // Check expectations
    UNITY_TEST_ASSERT_NULL(C, __LINE__, "Expected NULL!");

    // Cleanup
    for (int i = 0; i < 2; i++)
    {
        free(A[i]);
        free(B[i]);
    }
    free(A);
    free(B);
}

// Build a rows x cols matrix (each row malloc'd) from flat row-major data
static float **make_matrix(const float *data, int rows, int cols)
{
    float **M = (float **)malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++)
    {
        M[i] = (float *)malloc(cols * sizeof(float));
        for (int j = 0; j < cols; j++)
        {
            M[i][j] = data[i * cols + j];
        }
    }
    return M;
}

static void free_matrix(float **M, int rows)
{
    for (int i = 0; i < rows; i++)
    {
        free(M[i]);
    }
    free(M);
}

// Multiplies flat A (M x K) by flat B (K x N), checks against flat expected (M x N), and frees everything
static void run_matmul_case(const float *aData, int M, int K, const float *bData, int N, const float *expectedData)
{
    float **A = make_matrix(aData, M, K);
    float **B = make_matrix(bData, K, N);
    float **expected = make_matrix(expectedData, M, N);

    float **C = matmul(A, B, M, K, K, N);
    TEST_ASSERT_NOT_NULL(C);
    assert_float_array_equal_matmul(expected, C, M, N);
    free_matrix(C, M);

    // Every hand-checked case also runs through the blocking version
    C = matmul_blocking(A, B, M, K, K, N);
    TEST_ASSERT_NOT_NULL(C);
    assert_float_array_equal_matmul(expected, C, M, N);

    free_matrix(A, M);
    free_matrix(B, K);
    free_matrix(expected, M);
    free_matrix(C, M);
}

// Primer example: [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]
void test_matmul_primer_example(void)
{
    float A[] = {1, 2,
                 3, 4};
    float B[] = {5, 6,
                 7, 8};
    float expected[] = {19, 22,
                        43, 50};
    run_matmul_case(A, 2, 2, B, 2, expected);
}

// Non-square: (2x3) x (3x2) = 2x2
void test_matmul_non_square(void)
{
    float A[] = {1, 2, 3,
                 4, 5, 6};
    float B[] = {7, 8,
                 9, 10,
                 11, 12};
    float expected[] = {58, 64,
                        139, 154};
    run_matmul_case(A, 2, 3, B, 2, expected);
}

// A x I = A
void test_matmul_identity(void)
{
    float A[] = {2, -1, 0.5f,
                 3, 4, -2,
                 0, 7, 1};
    float I[] = {1, 0, 0,
                 0, 1, 0,
                 0, 0, 1};
    run_matmul_case(A, 3, 3, I, 3, A);
}

// (1x4) x (4x1) = 1x1 dot product
void test_matmul_dot_product(void)
{
    float A[] = {1, 2, 3, 4};
    float B[] = {5, 6, 7, 8};
    float expected[] = {70};
    run_matmul_case(A, 1, 4, B, 1, expected);
}

// (2x1) x (1x3) = 2x3 outer product: output larger than both inputs
void test_matmul_outer_product(void)
{
    float A[] = {1,
                 2};
    float B[] = {3, 4, 5};
    float expected[] = {3, 4, 5,
                        6, 8, 10};
    run_matmul_case(A, 2, 1, B, 3, expected);
}

// (3x2) x (3x2): inner dimensions 2 != 3, must return NULL
void test_matmul_incompatible_rectangular(void)
{
    float data[] = {1, 2, 3, 4, 5, 6};
    float **A = make_matrix(data, 3, 2);
    float **B = make_matrix(data, 3, 2);

    float **C = matmul(A, B, 3, 2, 3, 2);
    UNITY_TEST_ASSERT_NULL(C, __LINE__, "Expected NULL!");
    C = matmul_blocking(A, B, 3, 2, 3, 2);
    UNITY_TEST_ASSERT_NULL(C, __LINE__, "Expected NULL!");

    free_matrix(A, 3);
    free_matrix(B, 3);
}

// Pseudo-random matrix with small integer values, so float sums are exact
// regardless of the order the blocked loops accumulate them in
static float **make_int_matrix(int rows, int cols, unsigned int seed)
{
    float **M = (float **)malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++)
    {
        M[i] = (float *)malloc(cols * sizeof(float));
        for (int j = 0; j < cols; j++)
        {
            seed = seed * 1103515245u + 12345u;
            M[i][j] = (float)((int)((seed >> 16) % 7) - 3);
        }
    }
    return M;
}

// matmul_blocking must equal matmul, including when no dimension is a multiple
// of the tile sizes (edge tiles) and when a matrix is smaller than one tile
static void run_blocking_vs_naive(int M, int K, int N)
{
    float **A = make_int_matrix(M, K, 1u + M);
    float **B = make_int_matrix(K, N, 7u + N);

    float **expected = matmul(A, B, M, K, K, N);
    float **C = matmul_blocking(A, B, M, K, K, N);
    assert_float_array_equal_matmul(expected, C, M, N);

    free_matrix(A, M);
    free_matrix(B, K);
    free_matrix(expected, M);
    free_matrix(C, M);
}

void test_matmul_blocking_square_matrices(void)
{
    float A[] = {1, 2,
                 3, 4};
    float B[] = {2, 0,
                 1, 2};
    float **Am = make_matrix(A, 2, 2);
    float **Bm = make_matrix(B, 2, 2);
    float expected[] = {4, 4,
                        10, 8};
    float **E = make_matrix(expected, 2, 2);

    float **C = matmul_blocking(Am, Bm, 2, 2, 2, 2);
    assert_float_array_equal_matmul(E, C, 2, 2);

    free_matrix(Am, 2);
    free_matrix(Bm, 2);
    free_matrix(E, 2);
    free_matrix(C, 2);
}

void test_matmul_blocking_matches_naive(void)
{
    run_blocking_vs_naive(1, 1, 1);
    run_blocking_vs_naive(5, 3, 4);       // smaller than one tile
    run_blocking_vs_naive(64, 64, 64);    // exact multiples of BLOCK_SIZE_J
    run_blocking_vs_naive(200, 50, 70);   // ragged edge tiles in every dimension
    run_blocking_vs_naive(171, 19, 33);   // one past a tile boundary in I, K and J
}


void test_matmul_quant(void)
{
    // Setup
    int num_quant_element_per_word = sizeof(unsigned int) * 8 / QUANT_PRECISION;
    int A_rows = 2, A_cols = 2;
    int B_rows = num_quant_element_per_word * A_cols, B_cols = 1;

    unsigned int **A = (unsigned int **)malloc(A_rows * sizeof(unsigned int *));
    for (int i = 0; i < A_rows; i++)
    {
        A[i] = (unsigned int *)malloc(A_cols * sizeof(unsigned int));
    }
    A[0][0] = 0x01010101;
    A[0][1] = 0x01010101;
    A[1][0] = 0x02020202;
    A[1][1] = 0x02020202;

    int **B = (int **)malloc(B_rows * sizeof(int *));
    for (int i = 0; i < B_rows; i++)
    {
        B[i] = (int *)malloc(B_cols * sizeof(int));
    }
    for (int i = 0; i < B_rows; i++)
    {
        for (int j = 0; j < B_cols; j++)
        {
            B[i][j] = i;
        }
    }

    float **expected = (float **)malloc(A_rows * sizeof(float *));
    for (int i = 0; i < A_rows; i++)
    {
        expected[i] = (float *)malloc(B_cols * sizeof(float));
    }
    expected[0][0] = 28.0f;
    expected[1][0] = 56.0f;

    // Run function under test
    float **C = matmul_quant(A, B, A_rows, A_cols, B_rows, B_cols);

    // Check expectations
    assert_float_array_equal_matmul(expected, C, A_rows, B_cols);

    // Cleanup
    for (int i = 0; i < A_rows; i++)
    {
        free(A[i]);
        free(C[i]);
        free(expected[i]);
    }
    for (int i = 0; i < B_rows; i++)
    {
        free(B[i]);
    }
    free(A);
    free(B);
    free(C);
    free(expected);
}

// Handout example: A[0][0] = 0x01020304 packs [1, 2, 3, 4] (most-significant
// byte first), so with B[0:4][0] = [a, b, c, d], C[0][0] = 1a + 2b + 3c + 4d.
// Distinct bytes catch a reversed byte order, which the symmetric
// 0x01010101 test above cannot.
void test_matmul_quant_byte_order(void)
{
    unsigned int a_row0[] = {0x01020304, 0xFF00800A};
    unsigned int *A[] = {a_row0};
    // B is 8 x 2: column 0 = [10, 20, 30, 40, 1, 2, 3, 4], column 1 = [1, -1, 1, -1, -1, 1, -1, 1]
    int b_data[8][2] = {{10, 1}, {20, -1}, {30, 1}, {40, -1}, {1, -1}, {2, 1}, {3, -1}, {4, 1}};
    int *B[8];
    for (int i = 0; i < 8; i++)
    {
        B[i] = b_data[i];
    }

    float **C = matmul_quant(A, B, 1, 2, 8, 2);
    TEST_ASSERT_NOT_NULL(C);
    // col 0: (1*10 + 2*20 + 3*30 + 4*40) + (255*1 + 0*2 + 128*3 + 10*4) = 300 + 679
    TEST_ASSERT_EQUAL_FLOAT(979.0f, C[0][0]);
    // col 1: (1 - 2 + 3 - 4) + (-255 + 0 - 128 + 10) = -2 - 373
    TEST_ASSERT_EQUAL_FLOAT(-375.0f, C[0][1]);

    free(C[0]);
    free(C);
}

// Wrong B_rows (must be 4 * A_cols) returns NULL
void test_matmul_quant_incompatible_dimensions(void)
{
    unsigned int a_row0[] = {0x01020304};
    unsigned int *A[] = {a_row0};
    int b_data[3][1] = {{1}, {2}, {3}};
    int *B[] = {b_data[0], b_data[1], b_data[2]};

    UNITY_TEST_ASSERT_NULL(matmul_quant(A, B, 1, 1, 3, 1), __LINE__, "Expected NULL!");
}
