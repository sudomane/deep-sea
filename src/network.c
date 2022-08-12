/**
 * @file    network.c
 * @author  Philippe Bouchet (philippe.bouchet@epita.fr)
 * @brief   Neural network API implementation.
 *          Internal API functions denoted with underscore preceding function name.
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "network.h"

#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"

/* NETWORK INTERNAL API */

/**
 * @brief Dynamic allocation of network layers
 * 
 * @param net Neural network struct
 */
static void _net_alloc_layers(network_t* net)
{
    net->a = calloc(net->L, sizeof(matrix_t*));
    net->z = calloc(net->L, sizeof(matrix_t*));
    net->w = calloc(net->L, sizeof(matrix_t*));
    net->b = calloc(net->L, sizeof(matrix_t*));
    net->delta = calloc(net->L, sizeof(matrix_t*));

    net->X = m_init(1, net->input_size);
    net->y = m_init(1, net->output_size);

    net->a[0] = m_init(1, net->hidden_size);
    net->z[0] = m_init(1, net->hidden_size);
    net->b[0] = m_init(1, net->hidden_size);
    net->delta[0] = m_init(1, net->hidden_size);

    net->w[0] = m_init(net->input_size, net->hidden_size);

    for (size_t l = 1; l < net->L - 1; l++)
    {
        net->a[l] = m_init(1, net->hidden_size);
        net->z[l] = m_init(1, net->hidden_size);
        net->b[l] = m_init(1, net->hidden_size);
        net->delta[l] = m_init(1, net->hidden_size);

        net->w[l] = m_init(net->hidden_size, net->hidden_size);
    }

    net->a[net->L - 1] = m_init(1, net->output_size);
    net->z[net->L - 1] = m_init(1, net->output_size);
    net->b[net->L - 1] = m_init(1, net->output_size);
    net->delta[net->L - 1] = m_init(1, net->output_size);

    net->w[net->L - 1] = m_init(net->hidden_size, net->output_size);
}

/**
 * @brief Free network layers
 * 
 * @param net Neural network struct
 */
static void _net_free_layers(network_t* net)
{
    m_free(net->X);
    m_free(net->y);
    
    for (size_t l = 0; l < net->L; l++)
    {
        m_free(net->a[l]);
        m_free(net->z[l]);
        m_free(net->b[l]);
        m_free(net->delta[l]);
        m_free(net->w[l]);
    }
    
    free(net->a);
    free(net->z);
    free(net->w);
    free(net->b);
    free(net->delta);
}

/**
 * @brief Randomizes network weights and biases
 * 
 * @param net Neural network struct
 */
static void _net_init_layers(network_t* net)
{
    for (size_t l = 0; l < net->L; l++)
    {
        m_fill(net->w[l], normalized_rand);
        m_fill(net->b[l], normalized_rand);
    }
}

/**
 * @brief Feed forward algorithm
 * 
 * @param net Neural network struct
 */
static void _net_feed_forward(network_t* net)
{
    m_mul(net->X, net->w[0], net->z[0]);
    m_add(net->z[0], net->b[0], net->z[0]);
    m_apply_dst(net->z[0], sigmoid, net->a[0]);

    for (size_t l = 1; l < net->L; l++)
    {
        m_mul(net->a[l-1], net->w[l], net->z[l]);
        m_add(net->z[l], net->b[l], net->z[l]);
        m_apply_dst(net->z[l], sigmoid, net->a[l]);
    }
}

/**
 * @brief Backpropagation algorithm
 * 
 * @param net Neural network struct
 */
static void _net_backprop(network_t* net)
{
    m_sub(net->a[net->L-1], net->y, net->delta[net->L-1]);
    matrix_t* d_zL = m_apply(net->z[net->L-1], d_sigmoid);
    m_hadamard(net->delta[net->L-1], d_zL, net->delta[net->L-1]);
    m_free(d_zL);

    for (int l = net->L-2; l >= 0; l--)
    {
        matrix_t* weights_trans = m_transpose(net->w[l+1]);
        m_mul(net->delta[l+1], weights_trans, net->delta[l]);

        matrix_t* d_zl = m_apply(net->z[l], d_sigmoid);
        m_hadamard(net->delta[l], d_zl, net->delta[l]);

        m_free(d_zl);
        m_free(weights_trans);
    }
}

/**
 * @brief Update network weights with delta layer
 * 
 * @param net Neural network struct
 */
static void _net_update_weights(network_t* net)
{
    double lr = 1;

    for (size_t l = net->L - 1; l > 0; l--)
    {
        matrix_t* a_trans = m_transpose(net->a[l-1]);
        matrix_t* w_copy = m_copy(net->w[l]);
        
        m_scalar_mul(net->w[l], lr, net->w[l]);
        m_mul(a_trans, net->delta[l], net->w[l]);
        m_sub(w_copy, net->w[l], net->w[l]);

        m_free(a_trans);
        m_free(w_copy);
    }
}

/**
 * @brief Update network biases with delta layer
 * 
 * @param net Neural network struct
 */
static void _net_update_bias(network_t* net)
{
    double lr = 0.01;

    for (size_t l = net->L - 1; l > 0; l--)
    {
        m_scalar_mul(net->delta[l], lr, net->delta[l]);
        m_sub(net->b[l], net->delta[l], net->b[l]);
    }
}

/**
 * @brief Initialize the network's input layer with data in X
 * 
 * @param net Neural network struct
 * @param X Array containing input_size amount of data
 */
static void _net_init_X(network_t* net, double* X)
{
    for(size_t i = 0; i < net->input_size; i++)
        net->X->array[i] = X[i];
}    

/**
 * @brief Initilaize the network's expected output with data in y
 * 
 * @param net Neural network struct
 * @param y Array containing output_size amount of data
 */
static void _net_init_y(network_t* net, double* y)
{
    for(size_t i = 0; i < net->output_size; i++)
        net->y->array[i] = y[i];
}

/* NETWORK PUBLIC API */

/**
 * @brief Initialize the network with desired parameters
 * 
 * @param input_size Number of neurons in the input layer
 * @param hidden_size Number of neurons in the hidden layer
 * @param output_size Number of neurons in the output layer
 * @param L Number of layers in the network, excluding the input layer
 * @return network_t* Pointer to the initialized neural network struct
 */
network_t* net_init(size_t input_size, size_t hidden_size, size_t output_size, size_t L)
{
    network_t* net = malloc(sizeof(network_t));

    net->input_size = input_size;
    net->hidden_size = hidden_size;
    net->output_size = output_size;
    net->L = L;

    _net_alloc_layers(net);
    _net_init_layers(net);

    return net;
}

/**
 * @brief Free the network
 * 
 * @param net Neural network struct
 */
void net_free(network_t* net)
{
    _net_free_layers(net);
    free(net);

    net = NULL;
}

/**
 * @brief Displays the network's various parameters, and their contents
 * 
 * @param net Neural network struct
 */
void net_display(network_t* net)
{
    printf("Input layer:\n");
    m_display(net->X);

    printf("Z-Activation:\n");
    for (size_t l = 0; l < net->L; l++)
        m_display(net->z[l]);

    printf("Activation:\n");
    for (size_t l = 0; l < net->L; l++)
        m_display(net->a[l]);

    printf("Weights:\n");
    for (size_t l = 0; l < net->L; l++)
        m_display(net->w[l]);

    printf("Bias:\n");
    for (size_t l = 0; l < net->L; l++)
        m_display(net->b[l]);

    printf("Delta:\n");
    for (size_t l = 0; l < net->L; l++)
        m_display(net->delta[l]);
}

/**
 * @brief Trains the network
 * 
 * @param net Neural network struct
 * @param epochs Amount of times the network should iterate on training
 */
void net_train(network_t* net, size_t epochs)
{
    double X_train[4][2] = {
		{0.f, 0.f},
		{0.f, 1.f},
		{1.f, 0.f},
		{1.f, 1.f}
	};

	double y_train[4][1] = {
		{0.f},
		{1.f},
		{1.f},
		{0.f}
	};

    size_t index = 1;

    for (size_t i = 0; i < epochs; i++)
    {
        size_t X_rand = rand() % 4;
        size_t y_rand = rand() % 4;

        double* X = X_train[index];
        double* y = y_train[index];

        _net_init_X(net, X);
        _net_init_y(net, y);

        _net_feed_forward(net);
        _net_backprop(net);
        _net_update_weights(net);
        _net_update_bias(net);
    }

    printf("Completed %zu epochs!\n", epochs);

    printf("\nInput: %f, %f\n\n", X_train[index][0], X_train[index][1]);
    
    printf("Prediction:\t%f\nExpected\t%f\n",
            net->a[net->L-1]->array[0],
            y_train[index][0]);
}