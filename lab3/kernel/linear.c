#include "linear.h"

float *linear(float *input, float **weights, float *biases, int inputSize, int outputSize)
{
    // Empty input or output: nothing to compute
    if (input == NULL || inputSize <= 0 || outputSize <= 0)
    {
        return NULL;
    }

    float *output = malloc(outputSize * sizeof(*output));

    // output[j] = biases[j] + sum_i input[i] * weights[j][i]
    for (int j = 0; j < outputSize; j++)
    {
        float sum = biases[j];
        for (int i = 0; i < inputSize; i++)
        {
            sum += input[i] * weights[j][i];
        }
        output[j] = sum;
    }

    return output;
}
