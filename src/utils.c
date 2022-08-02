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
    return sigmoid(x) * (1.f - sigmoid(x));
}