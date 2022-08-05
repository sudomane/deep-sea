#ifndef NETWORK_H
#define NETWORK_H

#define INPUT_SIZE 2
#define HIDDEN_SIZE 2
#define OUTPUT_SIZE 1

#define MAT_SIZE (INPUT_SIZE) > (HIDDEN_SIZE) ? (INPUT_SIZE) : (HIDDEN_SIZE)

#define N_HIDDEN_LAYER 1

#include "matrix.h"

typedef struct
{
    matrix_2D_t* activation; // Matrix containing activation values (input + hidden + output)
    matrix_2D_t* weights; // Matrix containing weights for each layer (hidden + output)
    matrix_2D_t* bias; // Matrix containing biases for each layer's neurons (hidden + output)

    double (* activation_function) (double);
    double learning_rate;
} network_t;

network_t* init_network(double* X, double (* activation_function) (double), double lr);
void free_network(network_t* network);
void summary(network_t* network, int verbose);

void init_weights(network_t* network);
void init_bias(network_t* network, size_t coef);
void init_input(network_t* network, double* input);

void feed_forward(network_t* network);
void back_propagation(network_t* network, double* y);

void train(network_t* network, double* y, size_t epochs);
void predict(network_t* network, double* X);

#endif // NETWORK_H