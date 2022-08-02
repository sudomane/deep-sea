#ifndef NETWORK_H
#define NETWORK_H

#define INPUT_SIZE 100
#define HIDDEN_SIZE 140
#define OUTPUT_SIZE 100

#define N_HIDDEN_LAYER 100

#include "matrix.h"

typedef struct
{
    matrix_2D_t* activation_network; // Matrix containing activation values
    matrix_2D_t* weights_network; // Matrix containing weights for each layer
    
    double bias[N_HIDDEN_LAYER + 1]; // Bias for every hidden layer + output layer
} network_t;

network_t* init_network();
void free_network(network_t* network);
void summary(network_t* network, int verbose);

void init_weights(network_t* network);
void init_bias(network_t* network);
void init_input(network_t* network, double* input);

void feed_forward(network_t* network);

// TODO: IMPLEMENT THESE!
void back_propagation(network_t* network_t);
void cost_function(network_t* network_t);


#endif // NETWORK_H