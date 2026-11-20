#include "conv.h"

// Im2col algorithm
float **im2col(float ***image, int numChannels, int imageSize, int kernelSize, int stride, int *outputSize)
{
    // No padding: output is outSize x outSize
    int outSize = (imageSize - kernelSize) / stride + 1;
    *outputSize = outSize;

    // Column matrix: one row per (channel, kernel row, kernel col) and one
    // column per output position, so conv = flattened kernels x cols
    int numRows = numChannels * kernelSize * kernelSize;
    int numCols = outSize * outSize;
    float **cols = malloc(numRows * sizeof(*cols));
    for (int r = 0; r < numRows; r++)
    {
        cols[r] = malloc(numCols * sizeof(**cols));
    }

    for (int c = 0; c < numChannels; c++)
    {
        for (int ki = 0; ki < kernelSize; ki++)
        {
            for (int kj = 0; kj < kernelSize; kj++)
            {
                int r = (c * kernelSize + ki) * kernelSize + kj;
                for (int i = 0; i < outSize; i++)
                {
                    for (int j = 0; j < outSize; j++)
                    {
                        cols[r][i * outSize + j] = image[c][i * stride + ki][j * stride + kj];
                    }
                }
            }
        }
    }

    return cols;
}

// Im2col algorithm's inverse
float ***col2im(float **result, int num_kernels, int conv_rows, int conv_cols)
{
    // result is num_kernels x (conv_rows * conv_cols); reshape each row to a 2D map
    float ***output = malloc(num_kernels * sizeof(*output));
    for (int f = 0; f < num_kernels; f++)
    {
        output[f] = malloc(conv_rows * sizeof(**output));
        for (int i = 0; i < conv_rows; i++)
        {
            output[f][i] = malloc(conv_cols * sizeof(***output));
            for (int j = 0; j < conv_cols; j++)
            {
                output[f][i][j] = result[f][i * conv_cols + j];
            }
        }
    }
    return output;
}

// Flatten kernels across all channels, matching im2col's row order
// (kernel_flatten above only reads channel 0)
static float **kernel_flatten_channels(float ****kernel, int num_kernels, int numChannels, int kernel_size)
{
    int len = numChannels * kernel_size * kernel_size;
    float **flattened = malloc(num_kernels * sizeof(*flattened));
    for (int f = 0; f < num_kernels; f++)
    {
        flattened[f] = malloc(len * sizeof(**flattened));
        for (int c = 0; c < numChannels; c++)
        {
            for (int i = 0; i < kernel_size; i++)
            {
                for (int j = 0; j < kernel_size; j++)
                {
                    flattened[f][(c * kernel_size + i) * kernel_size + j] = kernel[f][c][i][j];
                }
            }
        }
    }
    return flattened;
}

static void free_2d(float **m, int rows)
{
    for (int i = 0; i < rows; i++)
    {
        free(m[i]);
    }
    free(m);
}

float **kernel_flatten(float ****kernel, int num_kernels, int kernel_size)
{
    float **flattened_kernels = (float **)malloc(num_kernels * sizeof(float *));
    for (int i = 0; i < num_kernels; i++)
    {
        flattened_kernels[i] = (float *)malloc(kernel_size * kernel_size * sizeof(float));
    }

    for (int k = 0; k < num_kernels; k++)
    {
        for (int i = 0; i < kernel_size; i++)
        {
            for (int j = 0; j < kernel_size; j++)
            {
                flattened_kernels[k][i * kernel_size + j] = kernel[k][0][i][j];
            }
        }
    }

    return flattened_kernels;
}

// Basic convolution operation
float ***convolution(float ***image, int numChannels, float ****kernel, float *biasData, int numFilters, int inputSize, int kernelSize)
{
    // Stride 1, no padding
    int outputSize = inputSize - kernelSize + 1;

    // Allocate output [numFilters][outputSize][outputSize]
    float ***convOutput = malloc(numFilters * sizeof(*convOutput));
    for (int f = 0; f < numFilters; f++)
    {
        convOutput[f] = malloc(outputSize * sizeof(**convOutput));
        for (int i = 0; i < outputSize; i++)
        {
            convOutput[f][i] = malloc(outputSize * sizeof(***convOutput));
        }
    }

    for (int f = 0; f < numFilters; f++)
    {
        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                // Sum over all channels and the kernel window
                float sum = 0.0f;
                for (int c = 0; c < numChannels; c++)
                {
                    for (int ki = 0; ki < kernelSize; ki++)
                    {
                        for (int kj = 0; kj < kernelSize; kj++)
                        {
                            sum += image[c][i + ki][j + kj] * kernel[f][c][ki][kj];
                        }
                    }
                }
                // Add the filter's bias once, then apply ReLU
                convOutput[f][i][j] = relu(sum + biasData[f]);
            }
        }
    }

    return convOutput;
}

// Convolution with im2col algorithm
float ***convolution_im2col(float ***image, int numChannels, float ****kernel, float *biasData, int numFilters, int inputSize, int kernelSize, MatmulType matmul_type)
{
    int kLen = numChannels * kernelSize * kernelSize;

    // Flatten kernel: numFilters x kLen
    float **flatKernels = kernel_flatten_channels(kernel, numFilters, numChannels, kernelSize);

    // Apply im2col: kLen x (outputSize * outputSize), stride 1
    int outputSize;
    float **cols = im2col(image, numChannels, inputSize, kernelSize, 1, &outputSize);
    int numCols = outputSize * outputSize;

    // Apply matmul: numFilters x numCols
    float **result;
    if (matmul_type == MATMUL_THREAD)
    {
        result = matmul_thread(flatKernels, cols, numFilters, kLen, kLen, numCols);
    }
    else
    {
        result = matmul(flatKernels, cols, numFilters, kLen, kLen, numCols);
    }

    // Apply col2im: numFilters x outputSize x outputSize
    float ***convOutput = col2im(result, numFilters, outputSize, outputSize);

    // Add bias and apply ReLU
    for (int f = 0; f < numFilters; f++)
    {
        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                convOutput[f][i][j] = relu(convOutput[f][i][j] + biasData[f]);
            }
        }
    }

    // Cleanup
    free_2d(flatKernels, numFilters);
    free_2d(cols, kLen);
    free_2d(result, numFilters);

    return convOutput;
}