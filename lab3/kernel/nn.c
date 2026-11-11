#include "nn.h"

float *flatten(float ***input, int inputSize, int depth)
{
  // Empty input: nothing to flatten
  if (input == NULL || inputSize <= 0 || depth <= 0)
  {
    return NULL;
  }

  // Row-major order: channel, then row, then column
  float *output = malloc(depth * inputSize * inputSize * sizeof(*output));
  for (int c = 0; c < depth; c++)
  {
    for (int i = 0; i < inputSize; i++)
    {
      for (int j = 0; j < inputSize; j++)
      {
        output[c * inputSize * inputSize + i * inputSize + j] = input[c][i][j];
      }
    }
  }
  return output;
}

void destroyConvOutput(float ***convOutput, int convOutputSize)
{
    for (int i = 0; i < 32; i++)
    {
        for (int j = 0; j < convOutputSize; j++)
        {
            free(convOutput[i][j]);
        }
        free(convOutput[i]);
    }
    free(convOutput);
}

int forwardPass(float ***image, int numChannels, float ****conv1WeightsData, float **fc1WeightsData, float **fc2WeightsData, float *conv1BiasData, float *fc1BiasData, float *fc2BiasData)
{
    // 1. Perform the convolution operation

    // 2. Flatten the output

    // 3. Perform the fully connected operations

    // 4. Apply the final softmax activation

    // 5. Make predictions

    // Clean up the memory usage
}

int predict(float *probabilityVector, int numClasses)
{
    // Argmax; strict > keeps the first index on ties
    int maxIndex = 0;
    for (int i = 1; i < numClasses; i++)
    {
        if (probabilityVector[i] > probabilityVector[maxIndex])
        {
            maxIndex = i;
        }
    }
    return maxIndex;
}
