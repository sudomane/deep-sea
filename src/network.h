/**
 * @file    network.h
 * @author  Philippe Bouchet (philippe.bouchet@epita.fr)
 * @brief   Neural network API header.
 *          Public API functions denoted with "net" prefix.
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef NETWORK_H
#define NETWORK_H

#include "matrix.h"

typedef struct
{
    size_t L;           // n hidden layers + output layer
    size_t input_size;
    size_t hidden_size;
    size_t output_size;

    matrix_t* X;        // Input data
    matrix_t* y;        // Expected output
    
    matrix_t** a;       // Activated neurons layer
    matrix_t** z;       // Pre activated neurons layer
    matrix_t** w;       // Weights layer
    matrix_t** b;       // Biases layer
    matrix_t** delta;   // Error delta layer
} network_t;

network_t* net_init(size_t input_size, size_t hidden_size, size_t output_size, size_t L);
void net_free(network_t* net);

void net_display(network_t* net);
void net_train(network_t* net, size_t epochs);

void net_evaluate(network_t* net);

#endif // NETWORK_H