#include "network.h"

#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"

network_t* init_network(double* X, double (*activation_function) (double), double lr)
{
    network_t* network = calloc(1, sizeof(network_t));

    if (network == NULL)
        errx(-1, "Could not allocate enough memory to create network!");

    network->activation_function = activation_function;
    network->activation = init_matrix((HIDDEN_SIZE), (N_HIDDEN_LAYER) + 2);
    network->weights = init_matrix((HIDDEN_SIZE) * (HIDDEN_SIZE), (N_HIDDEN_LAYER) + 1);
    network->bias = init_matrix((HIDDEN_SIZE), (N_HIDDEN_LAYER) + 1);
    network->learning_rate = lr;

    fill_matrix(network->activation, -1.f);
    fill_matrix(network->weights, -1.f);
    fill_matrix(network->bias, -1.f);
    
    init_input(network, X);
    init_weights(network);
    init_bias(network, 1);

    return network;
}

void free_network(network_t* network)
{
    free_matrix(network->activation);
    free_matrix(network->weights);
    free_matrix(network->bias);
    free(network);
}

void summary(network_t* network, int verbose)
{
    printf("\n");
    printf("|\tINPUT SIZE:\t%d\t|\n", (INPUT_SIZE));
    printf("|\tHIDDEN SIZE:\t%d @ %d\t|\n", (HIDDEN_SIZE), (N_HIDDEN_LAYER));
    printf("|\tOUTPUT SIZE:\t%d\t|\n", (OUTPUT_SIZE));
    printf("\n");

    if (verbose)
    {
        size_t n_params = matrix_size(network->activation)
                        + matrix_size(network->bias)
                        + matrix_size(network->weights)
                        - (INPUT_SIZE);
        printf("Number of parameters: %zu\n", n_params);

        printf("Bias:\n");
        display_matrix(network->bias);
        
        printf("\nActivation network:\n");
        display_matrix(network->activation);
        
        printf("\nWeights network:\n");
        display_matrix(network->weights);
    }
}

void init_weights(network_t* network)
{
    // INPUT to 1st HIDDEN LAYER
    for (size_t i = 0; i < (INPUT_SIZE) * (HIDDEN_SIZE); i++)
    {
        set_at(network->weights, i, 0, normalized_rand());
    }

    // 1st HIDDEN LAYER to Nth HIDDEN LAYER
    for (size_t i = 0; i < (HIDDEN_SIZE) * (HIDDEN_SIZE); i++)
    {
        for (size_t j = 1; j < (N_HIDDEN_LAYER); j++)
        {
            set_at(network->weights, i, j, normalized_rand());
        }
    }

    // Nth HIDDEN LAYER to OUTPUT LAYER
    for (size_t i = 0; i < (HIDDEN_SIZE) * (OUTPUT_SIZE); i++)
    {
        set_at(network->weights, i, (N_HIDDEN_LAYER), normalized_rand());
    }
}

void init_bias(network_t* network, size_t coef)
{
    for (size_t i = 0; i < (HIDDEN_SIZE); i++)
    {
        for (size_t j = 0; j < (N_HIDDEN_LAYER); j++)
        {
            set_at(network->bias, i, j, normalized_rand() * coef);
        }
    }
    
    for (size_t i = 0; i < (OUTPUT_SIZE); i++)
    {
        set_at(network->bias, i, (N_HIDDEN_LAYER), normalized_rand() * coef);
    }
}

void init_input(network_t* network, double* input)
{
    for (size_t i = 0; i < (INPUT_SIZE); i++)
    {
        network->activation[0].array[i] = input[i];
    }
}

void feed_forward(network_t* network)
{
    // ITERATE OVER LAYERS
    for (size_t l = 1; l <= (N_HIDDEN_LAYER) + 1; l++)
    {
        size_t current_layer_size = (HIDDEN_SIZE);
        
        if (l == (N_HIDDEN_LAYER) + 1)
            current_layer_size = (OUTPUT_SIZE);

        // ITERATE OVER LAYER NEURONS
        for (size_t j = 0; j < current_layer_size; j++)
        {
            size_t previous_layer_size = (HIDDEN_SIZE);
            
            if ((l-1) == 0)
                previous_layer_size = (INPUT_SIZE);

            // z - Sum of all bias + weights
            double z = 0.f;            

            // ITERATE OVER PREVIOUS LAYER WEIGHTS
            for (size_t k = 0; k < previous_layer_size; k++)
            {
                // previous activation
                double x = get_at(network->activation, k, (l-1));

                // weight
                double w = get_at(network->weights, k + j * previous_layer_size, (l-1));

                z += x*w;
            }
        
            // b - Bias
            double b = get_at(network->bias, j, l-1);

            // a - Activation
            double a = network->activation_function(z - b   );
            set_at(network->activation, j, l, a);
        }
    }
}

// Still broken
void back_propagation(network_t* network, double* y)
{
    double output_error[(OUTPUT_SIZE)] = { 0 };
    size_t last_layer = (N_HIDDEN_LAYER);
    
    for (size_t j = 0; j < (OUTPUT_SIZE); j++)
    {
        for (size_t k = 0; k < (HIDDEN_SIZE); k++)
        {
            double w_o = get_at(network->weights, k + j * (HIDDEN_SIZE), last_layer);
            output_error[j] += (w_o - y[j]) * d_sigmoid(w_o);
        }
    }

    for (size_t j = 0; j < (OUTPUT_SIZE); j++)
    {
        for (size_t k = 0; k < (HIDDEN_SIZE); k++)
        {
            double w_o = get_at(network->weights, k + j * (HIDDEN_SIZE), last_layer);
            double input_neuron = get_at(network->activation, j, last_layer - 1);
            double new_weight = w_o - network->learning_rate * output_error[j] * input_neuron;
            
            set_at(network->weights, k + j * (HIDDEN_SIZE), last_layer, new_weight);
        }
    }

    // Weight layer index
    for (int l = (N_HIDDEN_LAYER); l > 0; l--)
    {
        size_t current_layer_size = (HIDDEN_SIZE);

        double hidden_error[(HIDDEN_SIZE)];

        size_t previous_layer_size = (HIDDEN_SIZE);
        if (l-1 == 0)
            previous_layer_size = (INPUT_SIZE);

        for (size_t j = 0; j < current_layer_size; j++)
        {         
            for (size_t k = 0; k < previous_layer_size; k++)
            {
                double w_h = get_at(network->weights, k + j * previous_layer_size, l-1);
                double current_y = get_at(network->activation, j, l);
                hidden_error[j] += (w_h - current_y) * d_sigmoid(w_h);
            }
        }

        for (size_t j = 0; j < current_layer_size; j++)
        {
            for (size_t k = 0; k < previous_layer_size; k++)
            {
                double w_h = get_at(network->weights, k + j * previous_layer_size, l-1);
                double input_neuron = get_at(network->activation, j, l-1);
                double new_weight = w_h - network->learning_rate * hidden_error[j] * input_neuron;
                
                set_at(network->weights, k + j * previous_layer_size, l-1, new_weight);
            }
        }
    }
}

void train(network_t* network, double* y, size_t epochs)
{
    for (size_t i = 0; i < epochs; i++)
    {
        //printf("EPOCH: %zu / %zu\n", i + 1, epochs);
        feed_forward(network);
        back_propagation(network, y);
    }
}

void predict(network_t* network, double* X)
{
    init_input(network, X);
    feed_forward(network);
    
    printf("Input X:\n");
    for (size_t i = 0; i < (INPUT_SIZE); i++)
    {
        printf("%f ", X[i]);
    }

    printf("\n\nPredicted Y:\n");
    for (size_t i = 0; i < (OUTPUT_SIZE); i++)
    {
        printf("%f ", get_at(network->activation, i, (N_HIDDEN_LAYER)));
    }

    printf("\n");
}