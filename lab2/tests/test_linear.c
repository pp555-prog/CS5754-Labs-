#include "unity/unity.h"
#include "../kernel/kernel.h"
#include "test_linear.h"


void test_linear_basic(void)
{
    float input[] = {1.0, 2.0, 3.0};
    float *weights[] = {(float[]){1.0, 2.0, 3.0}, (float[]){4.0, 5.0, 6.0}};
    float biases[] = {0.1, 0.2};
    float *output = linear(input, weights, biases, 3, 2);
    TEST_ASSERT_EQUAL_FLOAT(14.1, output[0]);
    TEST_ASSERT_EQUAL_FLOAT(32.2, output[1]);

    // Cleanup
    free(output);
}

void test_linear_basic2(void)
{
    float input[] = {1.0, 2.0, 3.0};
    float *weights[] = {(float[]){1.0, 2.0, 3.0}, (float[]){4.0, 5.0, 6.0}};
    float biases[] = {0.1, 0.2};
    float *output = linear(input, weights, biases, 3, 2);
    TEST_ASSERT_EQUAL_FLOAT(14.1, output[0]);
    TEST_ASSERT_EQUAL_FLOAT(32.2, output[1]);

    // Cleanup
    free(output);
}

// Add more test cases as needed

#define LINEAR_EPSILON 1e-5f

// Primer example x = [1, 2], W[in][out] = [[1, 0, 2], [3, 1, 0]] -> [7, 2, 2].
// linear() takes weights as [out][in], so the primer's grid is transposed here.
void test_linear_primer_example(void)
{
    float input[] = {1.0, 2.0};
    float *weights[] = {(float[]){1.0, 3.0}, (float[]){0.0, 1.0}, (float[]){2.0, 0.0}};
    float biases[] = {0.0, 0.0, 0.0};
    float *output = linear(input, weights, biases, 2, 3);
    TEST_ASSERT_FLOAT_WITHIN(LINEAR_EPSILON, 7.0, output[0]);
    TEST_ASSERT_FLOAT_WITHIN(LINEAR_EPSILON, 2.0, output[1]);
    TEST_ASSERT_FLOAT_WITHIN(LINEAR_EPSILON, 2.0, output[2]);

    // Cleanup
    free(output);
}

// Identity weights with zero bias: output equals input
void test_linear_identity(void)
{
    float input[] = {1.5, -2.0, 3.0};
    float *weights[] = {(float[]){1.0, 0.0, 0.0}, (float[]){0.0, 1.0, 0.0}, (float[]){0.0, 0.0, 1.0}};
    float biases[] = {0.0, 0.0, 0.0};
    float *output = linear(input, weights, biases, 3, 3);
    for (int i = 0; i < 3; i++)
    {
        TEST_ASSERT_FLOAT_WITHIN(LINEAR_EPSILON, input[i], output[i]);
    }

    // Cleanup
    free(output);
}

// Negative inputs, weights and biases; no ReLU inside linear, so negatives survive
void test_linear_negative_values(void)
{
    float input[] = {-1.0, 2.0};
    float *weights[] = {(float[]){0.5, -1.0}, (float[]){2.0, 3.0}};
    float biases[] = {1.0, -4.0};
    float *output = linear(input, weights, biases, 2, 2);
    TEST_ASSERT_FLOAT_WITHIN(LINEAR_EPSILON, -1.5, output[0]); // -0.5 - 2 + 1
    TEST_ASSERT_FLOAT_WITHIN(LINEAR_EPSILON, 0.0, output[1]);  // -2 + 6 - 4

    // Cleanup
    free(output);
}

// Many inputs to a single output
void test_linear_many_to_one(void)
{
    float input[] = {1.0, 2.0, 3.0, 4.0};
    float *weights[] = {(float[]){1.0, 1.0, 1.0, 1.0}};
    float biases[] = {0.5};
    float *output = linear(input, weights, biases, 4, 1);
    TEST_ASSERT_FLOAT_WITHIN(LINEAR_EPSILON, 10.5, output[0]);

    // Cleanup
    free(output);
}
