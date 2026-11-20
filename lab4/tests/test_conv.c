#include "unity/unity.h"
#include "../kernel/kernel.h"
#include "../utils/data_utils.h"
#include "test_conv.h"


#define EPSILON 0.000001f

void free_image(float*** image, int numChannels, int inputSize) {
    for (int c = 0; c < numChannels; c++) {
        for (int i = 0; i < inputSize; i++) {
            free(image[c][i]);
        }
        free(image[c]);
    }
    free(image);
}

void free_kernel(float**** kernel, int numFilters, int kernelSize) {
    for (int i = 0; i < numFilters; i++) {
        for (int j = 0; j < 1; j++) {
            for (int k = 0; k < kernelSize; k++) {
                free(kernel[i][j][k]);
            }
            free(kernel[i][j]);
        }
        free(kernel[i]);
    }
    free(kernel);
}

void free_convOutput(float*** convOutput, int numFilters, int outputSize) {
    for (int i = 0; i < numFilters; i++) {
        for (int j = 0; j < outputSize; j++) {
            free(convOutput[i][j]);
        }
        free(convOutput[i]);
    }
    free(convOutput);
}


void assert_float_array_equal_conv(float ***expected, float ***actual, int depth, int rows, int cols)
{
    for (int i = 0; i < depth; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            for (int k = 0; k < cols; k++)
            {
                UNITY_TEST_ASSERT_FLOAT_WITHIN(EPSILON, expected[i][j][k], actual[i][j][k], __LINE__, "Arrays Not Equal!");
            }
        }
    }
}

void test_conv(void)
{
    // Setup
    float image_data[1][5][5] = {
        {
            {0, 0, 0, 0, 0},
            {0, 1, 1, 1, 0},
            {0, 1, 0, 1, 0},
            {0, 1, 1, 1, 0},
            {0, 0, 0, 0, 0}
        }
    };
    int numChannels = 1;
    float ***image = init_image(image_data, 5, numChannels);


    float kernel_data[1][1][3][3] = {
        {
            {
                {1, 0, 1},
                {2, 0, 2},
                {1, 0, 1}
            }
        }
    };
    int numFilters = 1;
    int kernelSize = 3;
    float ****kernel = init_kernel(kernel_data, numFilters, kernelSize);

    // Initialize the bias
    float *biasData = (float *)malloc(1 * sizeof(float));
    biasData[0] = 0;

    float ***expected = (float ***)malloc(1 * sizeof(float **));
    for(int i = 0; i < 1; i++) {
        expected[i] = (float **)malloc(3 * sizeof(float *));
        for(int j = 0; j < 3; j++) {
            expected[i][j] = (float *)malloc(3 * sizeof(float));
        }
    }
    
    // Initialize the expected result
    float expected_values[1][3][3] = {
        {
            {2, 6, 2},
            {2, 8, 2},
            {2, 6, 2}
        }
    };
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                expected[i][j][k] = expected_values[i][j][k];
            }
        }
    }

    // Run function under test
    float ***convOutput = convolution(image, numChannels, kernel, biasData, 1, 5, 3);

    // Check expectations
    assert_float_array_equal_conv(expected, convOutput, 1, 3, 3);

    // Cleanup
    free_image(image, numChannels, 5);
    free_kernel(kernel, 1, 3);
    free(biasData);
    for(int i = 0; i < 1; i++) {
        for(int j = 0; j < 3; j++) {
            free(expected[i][j]);
        }
        free(expected[i]);
    }
    free(expected);
    free_convOutput(convOutput, numFilters, kernelSize);
}

// Generic helpers: build arrays of any shape from flat row-major data.
// (init_image/init_kernel only support 5x5 images and 1-channel 3x3 kernels.)

static float ***make_image(const float *data, int numChannels, int inputSize)
{
    float ***image = malloc(numChannels * sizeof(*image));
    for (int c = 0; c < numChannels; c++)
    {
        image[c] = malloc(inputSize * sizeof(**image));
        for (int i = 0; i < inputSize; i++)
        {
            image[c][i] = malloc(inputSize * sizeof(***image));
            for (int j = 0; j < inputSize; j++)
            {
                image[c][i][j] = data[(c * inputSize + i) * inputSize + j];
            }
        }
    }
    return image;
}

static float ****make_kernel(const float *data, int numFilters, int numChannels, int kernelSize)
{
    float ****kernel = malloc(numFilters * sizeof(*kernel));
    for (int f = 0; f < numFilters; f++)
    {
        kernel[f] = malloc(numChannels * sizeof(**kernel));
        for (int c = 0; c < numChannels; c++)
        {
            kernel[f][c] = malloc(kernelSize * sizeof(***kernel));
            for (int i = 0; i < kernelSize; i++)
            {
                kernel[f][c][i] = malloc(kernelSize * sizeof(****kernel));
                for (int j = 0; j < kernelSize; j++)
                {
                    kernel[f][c][i][j] = data[((f * numChannels + c) * kernelSize + i) * kernelSize + j];
                }
            }
        }
    }
    return kernel;
}

static void free_kernel_c(float ****kernel, int numFilters, int numChannels, int kernelSize)
{
    for (int f = 0; f < numFilters; f++)
    {
        for (int c = 0; c < numChannels; c++)
        {
            for (int i = 0; i < kernelSize; i++)
            {
                free(kernel[f][c][i]);
            }
            free(kernel[f][c]);
        }
        free(kernel[f]);
    }
    free(kernel);
}

static void assert_conv_output(const float *expected, float ***actual, int numFilters, int outputSize)
{
    for (int f = 0; f < numFilters; f++)
    {
        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                UNITY_TEST_ASSERT_FLOAT_WITHIN(EPSILON, expected[(f * outputSize + i) * outputSize + j], actual[f][i][j], __LINE__, "Arrays Not Equal!");
            }
        }
    }
}

// Runs convolution on flat inputs, checks against flat expected output, and frees everything.
// Each case is checked with the direct convolution and with im2col (naive and multithreaded matmul).
static void run_conv_case(const float *imageData, int numChannels, int inputSize,
                          const float *kernelData, int numFilters, int kernelSize,
                          float *biasData, const float *expected)
{
    float ***image = make_image(imageData, numChannels, inputSize);
    float ****kernel = make_kernel(kernelData, numFilters, numChannels, kernelSize);
    int outputSize = inputSize - kernelSize + 1;

    float ***convOutput = convolution(image, numChannels, kernel, biasData, numFilters, inputSize, kernelSize);
    assert_conv_output(expected, convOutput, numFilters, outputSize);
    free_convOutput(convOutput, numFilters, outputSize);

    convOutput = convolution_im2col(image, numChannels, kernel, biasData, numFilters, inputSize, kernelSize, MATMUL_BASE);
    assert_conv_output(expected, convOutput, numFilters, outputSize);
    free_convOutput(convOutput, numFilters, outputSize);

    convOutput = convolution_im2col(image, numChannels, kernel, biasData, numFilters, inputSize, kernelSize, MATMUL_THREAD);
    assert_conv_output(expected, convOutput, numFilters, outputSize);
    free_convOutput(convOutput, numFilters, outputSize);

    free_image(image, numChannels, inputSize);
    free_kernel_c(kernel, numFilters, numChannels, kernelSize);
}

// Handout example: 2-channel 4x4 image, one 2x2 filter, bias 2
void test_conv_handout_example(void)
{
    float imageData[2 * 4 * 4] = {
        // Channel 0
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11,
        12, 13, 14, 15,
        // Channel 1
        1, 1, 3, 3,
        5, 5, 7, 7,
        9, 9, 11, 11,
        13, 13, 15, 15};
    float kernelData[1 * 2 * 2 * 2] = {
        1, 1, 1, 1,  // Filter 0, channel 0
        2, 2, 2, 2}; // Filter 0, channel 1
    float bias[1] = {2};
    float expected[1 * 3 * 3] = {
        36, 48, 60,
        84, 96, 108,
        132, 144, 156};

    run_conv_case(imageData, 2, 4, kernelData, 1, 2, bias, expected);
}

// Negative weights push some outputs below zero, which ReLU must clip to 0
void test_conv_relu_clips_negatives(void)
{
    float imageData[1 * 3 * 3] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9};
    float kernelData[1 * 1 * 2 * 2] = {
        -1, 0,
        0, 0};
    float bias[1] = {2};
    // 2 - image[i][j] = {1, 0, -2, -3} -> ReLU
    float expected[1 * 2 * 2] = {
        1, 0,
        0, 0};

    run_conv_case(imageData, 1, 3, kernelData, 1, 2, bias, expected);
}

// Three filters, each with its own bias: checks output depth and per-filter bias
void test_conv_multiple_filters(void)
{
    float imageData[1 * 3 * 3] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9};
    float kernelData[3 * 1 * 2 * 2] = {
        1, 1, 1, 1,  // Filter 0: window sum
        1, 1, 1, 1,  // Filter 1: window sum, large negative bias
        0, 0, 0, 1}; // Filter 2: bottom-right pixel
    float bias[3] = {0, -20, 0.5f};
    float expected[3 * 2 * 2] = {
        // Filter 0: 12, 16, 24, 28
        12, 16,
        24, 28,
        // Filter 1: -8, -4, 4, 8 -> ReLU
        0, 0,
        4, 8,
        // Filter 2: 5, 6, 8, 9 plus 0.5
        5.5f, 6.5f,
        8.5f, 9.5f};

    run_conv_case(imageData, 1, 3, kernelData, 3, 2, bias, expected);
}

// 1x1 kernel: output is a per-pixel weighted sum of the channels
void test_conv_1x1_kernel(void)
{
    float imageData[2 * 2 * 2] = {
        // Channel 0
        1, 2,
        3, 4,
        // Channel 1
        1, 1,
        1, 1};
    float kernelData[1 * 2 * 1 * 1] = {2, -1};
    float bias[1] = {0};
    // 2 * ch0 - ch1
    float expected[1 * 2 * 2] = {
        1, 3,
        5, 7};

    run_conv_case(imageData, 2, 2, kernelData, 1, 1, bias, expected);
}

// Kernel as large as the image: a single output value
void test_conv_kernel_covers_image(void)
{
    float imageData[1 * 3 * 3] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9};
    float kernelData[1 * 1 * 3 * 3] = {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1};
    float bias[1] = {1};
    float expected[1] = {46};

    run_conv_case(imageData, 1, 3, kernelData, 1, 3, bias, expected);
}


void test_conv_im2col(void)
{
    // Setup
    float image_data[1][5][5] = {
        {
            {0, 0, 0, 0, 0},
            {0, 1, 1, 1, 0},
            {0, 1, 0, 1, 0},
            {0, 1, 1, 1, 0},
            {0, 0, 0, 0, 0}
        }
    };
    int numChannels = 1;
    float ***image = init_image(image_data, 5, numChannels);


    float kernel_data[1][1][3][3] = {
        {
            {
                {1, 0, 1},
                {2, 0, 2},
                {1, 0, 1}
            }
        }
    };
    int numFilters = 1;
    int kernelSize = 3;
    float ****kernel = init_kernel(kernel_data, numFilters, kernelSize);

    // Initialize the bias
    float *biasData = (float *)malloc(1 * sizeof(float));
    biasData[0] = 0;

    float ***expected = (float ***)malloc(1 * sizeof(float **));
    for(int i = 0; i < 1; i++) {
        expected[i] = (float **)malloc(3 * sizeof(float *));
        for(int j = 0; j < 3; j++) {
            expected[i][j] = (float *)malloc(3 * sizeof(float));
        }
    }
    
    // Initialize the expected result
    float expected_values[1][3][3] = {
        {
            {2, 6, 2},
            {2, 8, 2},
            {2, 6, 2}
        }
    };
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                expected[i][j][k] = expected_values[i][j][k];
            }
        }
    }

    // Run function under test
    float ***convOutput = convolution_im2col(image, numChannels, kernel, biasData, 1, 5, 3, MATMUL_BASE);

    // Check expectations
    assert_float_array_equal_conv(expected, convOutput, 1, 3, 3);

    // Cleanup
    free_image(image, numChannels, 5);
    free_kernel(kernel, 1, 3);
    free(biasData);
    for(int i = 0; i < 1; i++) {
        for(int j = 0; j < 3; j++) {
            free(expected[i][j]);
        }
        free(expected[i]);
    }
    free(expected);
    free_convOutput(convOutput, numFilters, kernelSize);
}