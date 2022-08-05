#include <math.h>
#include <stdlib.h>

#include "utils.h"

double normalized_rand()
{
    return (double) rand()/ (double) RAND_MAX;
}

double sigmoid(double x)
{
    return (1.f / (1.f + exp(-x)));
}

double d_sigmoid(double x)
{
    return x * (1.f - x);
}

double relu(double x)
{
    return x > 0.f ? x : 0.f;
}

double d_relu(double x)
{
    return x > 0.f ? 1.f : 0.f;
}