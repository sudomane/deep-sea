#include "network.h"

#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"

network_t* init_network()
{
    network_t* network = calloc(1, sizeof(network_t));

    if (network == NULL)
        errx(-1, "Could not allocate enough memory to create network!");

    network->activation_network = init_matrix((HIDDEN_SIZE), (N_HIDDEN_LAYER) + 2);
    network->weights_network = init_matrix((HIDDEN_SIZE) * (HIDDEN_SIZE), (N_HIDDEN_LAYER) + 1);

    fill_matrix(network->activation_network, -1.f);
    fill_matrix(network->weights_network, -1.f);

    init_weights(network);
    init_bias(network);

    return network;
}

void free_network(network_t* network)
{
    free_matrix(network->activation_network);
    free_matrix(network->weights_network);
    free(network);
}

void display_network(network_t* network)
{
    printf("Bias:\n");
    for (size_t i = 0; i < N_HIDDEN_LAYER+1; i++)
    {
        printf("%f ", network->bias[i]);
    }
    printf("\nActivation network:\n");
    display_matrix(network->activation_network);
    
    printf("\nWeights network:\n");
    display_matrix(network->weights_network);
}

void init_weights(network_t* network)
{
    // INPUT to 1st HIDDEN LAYER
    for (size_t i = 0; i < (INPUT_SIZE) * (HIDDEN_SIZE); i++)
    {
        set_at(network->weights_network, i, 0, normalized_rand());
    }

    // 1st HIDDEN LAYER to Nth HIDDEN LAYER
    for (size_t i = 0; i < (HIDDEN_SIZE) * (HIDDEN_SIZE); i++)
    {
        for (size_t j = 1; j < (N_HIDDEN_LAYER); j++)
        {
            set_at(network->weights_network, i, j, normalized_rand());
        }
    }

    // Nth HIDDEN LAYER to OUTPUT LAYER
    for (size_t i = 0; i < (HIDDEN_SIZE) * (OUTPUT_SIZE); i++)
    {
        set_at(network->weights_network, i, (N_HIDDEN_LAYER), normalized_rand());
    }
}

void init_bias(network_t* network)
{
    for (size_t i = 0; i < (N_HIDDEN_LAYER) + 1; i++)
    {
        network->bias[i] = normalized_rand();
    }
}

void init_input(network_t* network, double* input)
{
    for (size_t i = 0; i < INPUT_SIZE; i++)
    {
        network->activation_network[0].array[i] = input[i];
    }
}

void feed_forward(network_t* network)
{
    // ITERATE OVER LAYERS
    for (size_t n = 1; n <= (N_HIDDEN_LAYER) + 1; n++)
    {
        size_t layer_neuron_size = (HIDDEN_SIZE);
        
        if (n == (N_HIDDEN_LAYER) + 1)
            layer_neuron_size = (OUTPUT_SIZE);

        // ITERATE OVER LAYER NEURONS
        for (size_t i = 0; i < layer_neuron_size; i++)
        {
            size_t previous_layer_size = (HIDDEN_SIZE);
            
            if ((n-1) == 0)
                previous_layer_size = (INPUT_SIZE);

            double activation = 0.f;            

            // ITERATE OVER PREVIOUS LAYER WEIGHTS (Eureka, nigga)
            for (size_t z = 0; z < previous_layer_size; z++)
            {
                double x = get_at(network->activation_network, z, (n-1));
                double w = get_at(network->weights_network, z + i * previous_layer_size, (n-1));

                activation += x*w;
            }
        
            activation = sigmoid(activation - network->bias[n-1]);
            set_at(network->activation_network, i, n, activation);
        }
    }
}