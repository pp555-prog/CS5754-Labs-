// Profiling driver for the Lab 1 and Lab 2 kernels.
//
// Usage: ./perf/profile <kernel> <size> <iters> [channels filters kernelSize]
//   relu, softmax : size = vector length
//   linear        : size = input size = output size (size x size weights)
//   matmul, matmul_blocking : size = N for (N x N) * (N x N)
//   conv          : size = image side; optional channels/filters/kernelSize (default 3 16 3)
//
// Inputs are allocated once; the kernel is then called <iters> times so the
// measured profile is dominated by the kernel, not setup/cleanup. Run with
// iters = 0 to profile setup alone and quantify its overhead.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "kernel/kernel.h"

static float rand_float(void)
{
    // Uniform in [-1, 1] so relu/conv see both signs
    return 2.0f * (float)rand() / RAND_MAX - 1.0f;
}

static float *rand_vector(int n)
{
    float *v = malloc(n * sizeof(*v));
    for (int i = 0; i < n; i++)
    {
        v[i] = rand_float();
    }
    return v;
}

static float **rand_matrix(int rows, int cols)
{
    float **m = malloc(rows * sizeof(*m));
    for (int i = 0; i < rows; i++)
    {
        m[i] = rand_vector(cols);
    }
    return m;
}

static void free_matrix(float **m, int rows)
{
    for (int i = 0; i < rows; i++)
    {
        free(m[i]);
    }
    free(m);
}

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        fprintf(stderr, "Usage: %s <relu|softmax|linear|matmul|matmul_blocking|conv> <size> <iters> [channels filters kernelSize]\n", argv[0]);
        return 1;
    }
    const char *name = argv[1];
    int size = atoi(argv[2]);
    long iters = atol(argv[3]);
    srand(5754); // fixed seed: identical inputs across trials

    float checksum = 0.0f;
    double start = 0.0, elapsed = 0.0;

    if (strcmp(name, "relu") == 0)
    {
        // relu() is scalar; apply it to every element into a separate output so the
        // input keeps its mix of signs across iterations (applyRelu works in place,
        // which would make every iteration after the first see only non-negatives).
        float *in = rand_vector(size);
        float *out = malloc(size * sizeof(*out));
        start = now_sec();
        for (long it = 0; it < iters; it++)
        {
            for (int i = 0; i < size; i++)
            {
                out[i] = relu(in[i]);
            }
        }
        elapsed = now_sec() - start;
        if (iters > 0)
            checksum = out[size / 2];
        free(in);
        free(out);
    }
    else if (strcmp(name, "softmax") == 0)
    {
        float *in = rand_vector(size);
        start = now_sec();
        for (long it = 0; it < iters; it++)
        {
            float *out = softmax(in, size);
            checksum = out[0];
            free(out);
        }
        elapsed = now_sec() - start;
        free(in);
    }
    else if (strcmp(name, "linear") == 0)
    {
        float *in = rand_vector(size);
        float **weights = rand_matrix(size, size);
        float *biases = rand_vector(size);
        start = now_sec();
        for (long it = 0; it < iters; it++)
        {
            float *out = linear(in, weights, biases, size, size);
            checksum = out[0];
            free(out);
        }
        elapsed = now_sec() - start;
        free(in);
        free_matrix(weights, size);
        free(biases);
    }
    else if (strcmp(name, "matmul") == 0 || strcmp(name, "matmul_blocking") == 0)
    {
        int blocking = strcmp(name, "matmul_blocking") == 0;
        float **A = rand_matrix(size, size);
        float **B = rand_matrix(size, size);
        start = now_sec();
        for (long it = 0; it < iters; it++)
        {
            float **C = blocking ? matmul_blocking(A, B, size, size, size, size)
                                 : matmul(A, B, size, size, size, size);
            checksum = C[0][0];
            free_matrix(C, size);
        }
        elapsed = now_sec() - start;
        free_matrix(A, size);
        free_matrix(B, size);
    }
    else if (strcmp(name, "conv") == 0)
    {
        int numChannels = argc > 4 ? atoi(argv[4]) : 3;
        int numFilters = argc > 5 ? atoi(argv[5]) : 16;
        int kernelSize = argc > 6 ? atoi(argv[6]) : 3;
        int outputSize = size - kernelSize + 1;

        float ***image = malloc(numChannels * sizeof(*image));
        for (int c = 0; c < numChannels; c++)
        {
            image[c] = rand_matrix(size, size);
        }
        float ****kernel = malloc(numFilters * sizeof(*kernel));
        for (int f = 0; f < numFilters; f++)
        {
            kernel[f] = malloc(numChannels * sizeof(**kernel));
            for (int c = 0; c < numChannels; c++)
            {
                kernel[f][c] = rand_matrix(kernelSize, kernelSize);
            }
        }
        float *bias = rand_vector(numFilters);

        start = now_sec();
        for (long it = 0; it < iters; it++)
        {
            float ***out = convolution(image, numChannels, kernel, bias, numFilters, size, kernelSize);
            checksum = out[0][0][0];
            for (int f = 0; f < numFilters; f++)
            {
                free_matrix(out[f], outputSize);
            }
            free(out);
        }
        elapsed = now_sec() - start;

        for (int c = 0; c < numChannels; c++)
        {
            free_matrix(image[c], size);
        }
        free(image);
        for (int f = 0; f < numFilters; f++)
        {
            for (int c = 0; c < numChannels; c++)
            {
                free_matrix(kernel[f][c], kernelSize);
            }
            free(kernel[f]);
        }
        free(kernel);
        free(bias);
    }
    else
    {
        fprintf(stderr, "Unknown kernel: %s\n", name);
        return 1;
    }

    printf("RESULT kernel=%s size=%d iters=%ld total_s=%.6f per_call_us=%.3f checksum=%f\n",
           name, size, iters, elapsed, iters > 0 ? elapsed / iters * 1e6 : 0.0, checksum);
    return 0;
}
