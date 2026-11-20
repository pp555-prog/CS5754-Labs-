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

    // Every hand-checked case also runs through the multithreaded version
    C = matmul_thread(A, B, M, K, K, N);
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

// A x I = A. (The identity is not named I: matrix_ops.h includes cblas.h,
// which pulls in <complex.h>, where I is a macro for the imaginary unit.)
void test_matmul_identity(void)
{
    float A[] = {2, -1, 0.5f,
                 3, 4, -2,
                 0, 7, 1};
    float identity[] = {1, 0, 0,
                        0, 1, 0,
                        0, 0, 1};
    run_matmul_case(A, 3, 3, identity, 3, A);
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
    C = matmul_thread(A, B, 3, 2, 3, 2);
    UNITY_TEST_ASSERT_NULL(C, __LINE__, "Expected NULL!");

    free_matrix(A, 3);
    free_matrix(B, 3);
}

/* ---------------- matmul_thread ---------------- */

// The provided square-matrix test, run on matmul_thread
void test_matmul_thread_square_matrices(void)
{
    float A[] = {1, 2,
                 3, 4};
    float B[] = {2, 0,
                 1, 2};
    float expected[] = {4, 4,
                        10, 8};
    float **Am = make_matrix(A, 2, 2);
    float **Bm = make_matrix(B, 2, 2);
    float **E = make_matrix(expected, 2, 2);

    float **C = matmul_thread(Am, Bm, 2, 2, 2, 2);
    TEST_ASSERT_NOT_NULL(C);
    assert_float_array_equal_matmul(E, C, 2, 2);

    free_matrix(Am, 2);
    free_matrix(Bm, 2);
    free_matrix(E, 2);
    free_matrix(C, 2);
}

// The provided incompatible-dimensions test, run on matmul_thread
void test_matmul_thread_incompatible_dimensions(void)
{
    float data[] = {1, 2, 3, 4, 5, 6};
    float **A = make_matrix(data, 2, 3);
    float **B = make_matrix(data, 2, 2);

    float **C = matmul_thread(A, B, 2, 3, 2, 2);
    UNITY_TEST_ASSERT_NULL(C, __LINE__, "Expected NULL!");

    free_matrix(A, 2);
    free_matrix(B, 2);
}

// Random matrix with values in [-1, 1)
static float **make_random_matrix(int rows, int cols, unsigned int seed)
{
    float **M = (float **)malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++)
    {
        M[i] = (float *)malloc(cols * sizeof(float));
        for (int j = 0; j < cols; j++)
        {
            seed = seed * 1103515245u + 12345u;
            M[i][j] = ((seed >> 8) & 0xFFFF) / 32768.0f - 1.0f;
        }
    }
    return M;
}

// matmul_thread must equal matmul on random non-square inputs. Each thread
// sums over k in the same order as matmul, so non-integer values match too.
static void run_thread_vs_naive(int M, int K, int N)
{
    float **A = make_random_matrix(M, K, 11u + M + K);
    float **B = make_random_matrix(K, N, 5u + N);

    float **expected = matmul(A, B, M, K, K, N);
    float **C = matmul_thread(A, B, M, K, K, N);
    TEST_ASSERT_NOT_NULL(C);
    assert_float_array_equal_matmul(expected, C, M, N);

    free_matrix(A, M);
    free_matrix(B, K);
    free_matrix(expected, M);
    free_matrix(C, M);
}

// Shapes cover: fewer rows than threads (1, 3), rows not divisible by the
// thread count (37, 100), rows of C shorter than a cache line (N = 1, 3),
// and rows that are not a multiple of a cache line (N = 29, 70).
static void run_thread_shapes(void)
{
    run_thread_vs_naive(1, 1, 1);
    run_thread_vs_naive(1, 64, 64);
    run_thread_vs_naive(3, 5, 3);
    run_thread_vs_naive(37, 53, 29);
    run_thread_vs_naive(64, 64, 64);
    run_thread_vs_naive(100, 80, 60);
    run_thread_vs_naive(50, 120, 70);
    run_thread_vs_naive(129, 17, 1);
}

// Default thread count (8)
void test_matmul_thread_matches_naive(void)
{
    unsetenv("MATMUL_THREADS");
    run_thread_shapes();
}

// Other thread counts, including more threads than rows
void test_matmul_thread_thread_counts(void)
{
    const char *counts[] = {"1", "2", "3", "7", "64"};
    for (int c = 0; c < 5; c++)
    {
        setenv("MATMUL_THREADS", counts[c], 1);
        run_thread_shapes();
    }
    unsetenv("MATMUL_THREADS");
}
