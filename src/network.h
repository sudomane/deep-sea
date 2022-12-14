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

#define NETWORK_FAILED_LOAD -1

#include "matrix.h"
#include "dataset.h"

typedef struct
__attribute__((packed, aligned(1)))
{
    size_t      L;           // n hidden layers + output layer
    size_t      input_size;
    size_t      hidden_size;
    size_t      output_size;

    size_t      batch_size;
    double      lr;          // Network learning rate

    matrix_t*   X;           // Input data
    matrix_t*   y;           // Expected output
    
    matrix_t**  a;           // Activated neurons layer
    matrix_t**  z;           // Pre activated neurons layer
    matrix_t**  w;           // Weights layer
    matrix_t**  b;           // Biases layer
    matrix_t**  delta;       // Error delta layer
    matrix_t**  grad_w;      // Cumulative batch gradient for weights
    matrix_t**  grad_b;      // Cumulative batch gradient for biases
} network_t;

network_t*  net_init(size_t L, size_t input_size,
                               size_t hidden_size,
                               size_t output_size,
                               size_t batch_size, double lr);

void        net_free(network_t* net);

network_t*  net_load(const char* path);
void        net_save(network_t* net, const char* dst);

void        net_summary(network_t* net);
void        net_train(network_t* net, dataset_t* dataset, size_t epochs);

void        net_evaluate(network_t* net, dataset_t* dataset);
void        net_predict(network_t* net, double* X, double* y);

#endif // NETWORK_H