#include "unity/unity.h"
#include "test_conv.h"
#include "test_nn.h"
#include "test_functional.h"
#include "test_linear.h"
#include "test_matrix_ops.h"

void setUp(void) {
    /* Code here will run before each test */
}

void tearDown(void) {
    /* Code here will run after each test */
}

int main(void) {
    UNITY_BEGIN();

    // Test conv (direct, and im2col with matmul and matmul_thread)
    RUN_TEST(test_conv);
    RUN_TEST(test_conv_handout_example);
    RUN_TEST(test_conv_relu_clips_negatives);
    RUN_TEST(test_conv_multiple_filters);
    RUN_TEST(test_conv_1x1_kernel);
    RUN_TEST(test_conv_kernel_covers_image);
    RUN_TEST(test_conv_im2col);

    // Test nn
    RUN_TEST(test_flatten_basic);
    RUN_TEST(test_predict_simple_array);
    RUN_TEST(test_predict_all_same_values);
    RUN_TEST(test_predict_mix_of_negatives_and_positives);

    // Test functional
    RUN_TEST(test_softmax_basic);
    RUN_TEST(test_relu);

    // Test linear
    RUN_TEST(test_linear_basic);
    RUN_TEST(test_linear_basic2);
    RUN_TEST(test_linear_primer_example);
    RUN_TEST(test_linear_identity);
    RUN_TEST(test_linear_negative_values);
    RUN_TEST(test_linear_many_to_one);

    // Test matrix_ops (hand-checked cases run on both matmul and matmul_thread)
    RUN_TEST(test_matmul_square_matrices);
    RUN_TEST(test_matmul_incompatible_dimensions);
    RUN_TEST(test_matmul_primer_example);
    RUN_TEST(test_matmul_non_square);
    RUN_TEST(test_matmul_identity);
    RUN_TEST(test_matmul_dot_product);
    RUN_TEST(test_matmul_outer_product);
    RUN_TEST(test_matmul_incompatible_rectangular);
    RUN_TEST(test_matmul_thread_square_matrices);
    RUN_TEST(test_matmul_thread_incompatible_dimensions);
    RUN_TEST(test_matmul_thread_matches_naive);
    RUN_TEST(test_matmul_thread_thread_counts);

    return UNITY_END();
}
