#ifndef TEST_MATRIX_OPS_H
#define TEST_MATRIX_OPS_H

void test_matmul_square_matrices(void);
void test_matmul_incompatible_dimensions(void);
void test_matmul_primer_example(void);
void test_matmul_non_square(void);
void test_matmul_identity(void);
void test_matmul_dot_product(void);
void test_matmul_outer_product(void);
void test_matmul_incompatible_rectangular(void);
void test_matmul_quant(void);
void test_matmul_quant_byte_order(void);
void test_matmul_quant_incompatible_dimensions(void);
void test_matmul_blocking_square_matrices(void);
void test_matmul_blocking_matches_naive(void);

#endif /* TEST_MATRIX_OPS_H */
