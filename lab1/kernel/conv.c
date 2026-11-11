#include "conv.h"

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
