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

/* Neural network public API */

#include "matrix.h"

typedef struct
{
    // Number of hidden layers + output layer
    size_t L;
    size_t input_size;
    size_t hidden_size;
    size_t output_size;

    matrix_t* X;
    matrix_t* y;
    
    matrix_t** a;
    matrix_t** z;
    matrix_t** w;
    matrix_t** b;
    matrix_t** delta;
} network_t;

network_t* net_init(size_t input_size, size_t hidden_size, size_t output_size, size_t L);
void net_free(network_t* net);

void net_display(network_t* net);
void net_train(network_t* net, size_t epochs);

void net_evaluate(network_t* net);

#endif // NETWORK_H