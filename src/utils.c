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

/**
 * @brief Returns a randomly generated normalized float between -1 and 1
 * 
 * @return double Value 
 */
double normalized_rand(void)
{
    return ((double) rand() / (double) RAND_MAX) * 2 - 1;
}

/**
 * @brief Sigmoid function
 * 
 * @param x 
 * @return double 
 */
double sigmoid(double x)
{
    return (1.f / (1.f + exp(-x)));
}

/**
 * @brief Sigmoid derivative function
 * 
 * @param x 
 * @return double 
 */
double d_sigmoid(double x)
{
    return sigmoid(x) * (1.f - sigmoid(x));
}

/**
 * @brief Relu function
 * 
 * @param x 
 * @return double 
 */
double relu(double x)
{
    return x > 0.f ? x : 0.f;
}

/**
 * @brief Relu derivative function
 * 
 * @param x 
 * @return double 
 */
double d_relu(double x)
{
    return x > 0.f ? 1.f : 0.f;
}