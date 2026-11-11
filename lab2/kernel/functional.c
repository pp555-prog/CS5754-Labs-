#include "functional.h"

float relu(float x)
{
    return x > 0.0f ? x : 0.0f;
}

void applyRelu(float *input, int inputSize)
{
    for (int i = 0; i < inputSize; i++)
    {
        input[i] = relu(input[i]);
    }
}

float *softmax(float *input, int inputSize)
{
    float *output = malloc(inputSize * sizeof(*output));

    // Find maximum of input vector
    float maxInput = -FLT_MAX;
    for (int i = 0; i < inputSize; i++)
    {
        if (input[i] > maxInput)
        {
            maxInput = input[i];
        }
    }

    // Compute exp of input - maxInput to avoid underflow
    float sum = 0.0f;
    for (int i = 0; i < inputSize; i++)
    {
        sum += expf(input[i] - maxInput);
    }

    // Normalise and apply log: log(e^(x - max) / sum) = (x - max) - log(sum)
    float logSum = logf(sum);
    for (int i = 0; i < inputSize; i++)
    {
        output[i] = (input[i] - maxInput) - logSum;
    }

    return output;
}
