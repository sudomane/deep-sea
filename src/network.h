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
    matrix_2D_t* activation_network; // Matrix containing activation values (input + hidden + output)
    matrix_2D_t* weights_network; // Matrix containing weights for each layer (hidden + output)
    matrix_2D_t* bias; // Matrix containing biases for each layer's neurons (hidden + output)
} network_t;

network_t* init_network();
void free_network(network_t* network);
void summary(network_t* network, int verbose);
size_t network_size(network_t* network);


void init_weights(network_t* network);
void init_bias(network_t* network, size_t coef);
void init_input(network_t* network, double* input);

void feed_forward(network_t* network);

// TODO: IMPLEMENT THESE!
void back_propagation(network_t* network, double* y);
double gradient_descent(network_t* network);
double cost_function(network_t* network, double* y);
void train(network_t* network, size_t epochs);

#endif // NETWORK_H