/**
 * @file    utils.c
 * @author  Philippe Bouchet (philippe.bouchet@epita.fr)
 * @brief   Utility functions implementation.
 *
 * @copyright Copyright (c) 2022
 * 
 */

#include "utils.h"

#include <math.h>
#include <stdlib.h>

double normalized_rand(void)
{
    return (double) rand() / (double) RAND_MAX;
}

double sigmoid(double x)
{
    return (1.f / (1.f + exp(-x)));
}

double d_sigmoid(double x)
{
    return sigmoid(x) * (1.f - sigmoid(x));
}

double relu(double x)
{
    return x > 0.f ? 0.01f * x : 0.f;
}

double d_relu(double x)
{
    return x > 0.f ? 0.01f : 0.f;
}