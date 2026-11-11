CC = gcc
CFLAGS = -I. -lm

HDF5_FLAGS = -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5
COMMON_HDRS = ./utils/data_utils.h ./kernel/conv.h ./kernel/matrix_ops.h ./kernel/linear.h ./kernel/functional.h ./kernel/nn.h
COMMON_SRC = ./utils/data_utils.c ./kernel/conv.c ./kernel/functional.c ./kernel/matrix_ops.c ./kernel/linear.c ./kernel/nn.c

# Unity test framework
UNITY_FILES = ./tests/unity/unity.c
TEST_FILES = ./tests/all_tests.c ./tests/test_conv.c ./tests/test_functional.c ./tests/test_linear.c ./tests/test_matrix_ops.c ./tests/test_nn.c
TEST_EXECUTABLES = $(patsubst %.c,%,$(TEST_FILES))

# Performance
MATMUL_TARGETS = matmul_naive matmul_sparse

.PHONY: test
test: all_tests

.PHONY: all_tests
all_tests:
	$(CC) -o tests/$@ $(TEST_FILES) $(UNITY_FILES) $(COMMON_SRC) $(CFLAGS) $(HDF5_FLAGS)
	./tests/$@

# Build only: profile on the ece5755 partition through srun + perf/on_core.sh,
# never on the login node (the template ran toplev here with core 0 hardcoded).
.PHONY : $(MATMUL_TARGETS)
$(MATMUL_TARGETS):
	$(CC) -o $@ ./perf/$@.c $(COMMON_SRC) $(CFLAGS) $(HDF5_FLAGS)

# Profiling driver (see perf/profile.c, perf/run_toplev.sh)
.PHONY: profile
profile:
	$(CC) -o perf/$@ ./perf/$@.c $(COMMON_SRC) $(CFLAGS) $(HDF5_FLAGS)

.PHONY: clean
clean:
	rm -f $(TEST_EXECUTABLES) $(GRADING_EXECUTABLES) $(GRADING_TESTS_OUTPUT) $(MATMUL_TARGETS) perf/profile
