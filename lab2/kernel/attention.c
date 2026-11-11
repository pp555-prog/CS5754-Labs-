#include "attention.h"
#include <stdio.h>
#include <math.h>

// Scaled dot-product attention
float **scaled_dot_product_attention(float **Q, float **K, float **V, int seqLength, int depth)
{
    // Output is seqLength x depth, each row allocated separately
    float **output = malloc(seqLength * sizeof(*output));
    for (int i = 0; i < seqLength; i++)
    {
        output[i] = calloc(depth, sizeof(**output));
    }

    float scale = 1.0f / sqrtf((float)depth);
    float *weights = malloc(seqLength * sizeof(*weights));

    for (int i = 0; i < seqLength; i++)
    {
        // scores[j] = (Q[i] . K[j]) / sqrt(depth), i.e. row i of Q K^T, scaled
        float maxScore = -INFINITY;
        for (int j = 0; j < seqLength; j++)
        {
            float dot = 0.0f;
            for (int d = 0; d < depth; d++)
            {
                dot += Q[i][d] * K[j][d];
            }
            weights[j] = dot * scale;
            if (weights[j] > maxScore)
            {
                maxScore = weights[j];
            }
        }

        // Row-wise softmax (plain probabilities, not the log-softmax that
        // softmax() returns); subtract the max for numerical stability
        float sum = 0.0f;
        for (int j = 0; j < seqLength; j++)
        {
            weights[j] = expf(weights[j] - maxScore);
            sum += weights[j];
        }
        for (int j = 0; j < seqLength; j++)
        {
            weights[j] /= sum;
        }

        // output[i] = sum_j weights[j] * V[j]
        for (int j = 0; j < seqLength; j++)
        {
            for (int d = 0; d < depth; d++)
            {
                output[i][d] += weights[j] * V[j][d];
            }
        }
    }

    free(weights);
    return output;
}
