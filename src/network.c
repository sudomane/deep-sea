/**
 * @file    network.c
 * @author  Philippe Bouchet (philippe.bouchet@epita.fr)
 * @brief   Neural network API implementation.
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "network.h"

#include <err.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "utils.h"

/* Internal API forward declaration */

static void     _net_alloc_layers(network_t* net);
static void     _net_free_layers(network_t* net);
static void     _net_init_layers(network_t* net);
static void     _net_feed_forward(network_t* net);
static void     _net_backprop(network_t* net);
static void     _net_mini_batch_gradient_descent(network_t* net);
static void     _net_update(network_t* net);
static void     _net_init_X(network_t* net, double* X);
static void     _net_init_y(network_t* net, double* y);
static double   _net_evaluate_prediction(network_t* net);
static void     _net_binarize_output(network_t* net, double threshold);


/* ==== NETWORK PUBLIC API ==== */


/**
 * @brief  Initialize the network with desired parameters
 * 
 * @param  L Number of layers in the network, excluding the input layer
 * @param  input_size Number of neurons in the input layer
 * @param  hidden_size Number of neurons in the hidden layer
 * @param  output_size Number of neurons in the output layer
 * @param  batch_size Amount of data to be used per epoch
 * @return network_t* Pointer to the initialized neural network struct
 */
network_t* net_init(size_t L, size_t input_size,
                              size_t hidden_size,
                              size_t output_size,
                              size_t batch_size, double lr)
{
    network_t* net = malloc(sizeof(network_t));

    net->L = L;
    net->input_size = input_size;
    net->hidden_size = hidden_size;
    net->output_size = output_size;
    net->batch_size = batch_size;
    net->lr = lr;

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
 * @brief  Load network from file and create network struct
 * 
 * @param  path Path to network parameters file
 * @return network_t* 
 */
network_t* net_load(const char* path)
{
    int fd = open(path, O_RDONLY);
    
    if (fd == -1)
    {
        errx(NETWORK_FAILED_LOAD,
             "NETWORK::ERROR::LOAD: "
             "Invalid path %s", path);
    }

    unsigned int signature;
    read(fd, &signature, sizeof(int));

    if (signature != 0xDEADBEEF)
    {
        errx(NETWORK_FAILED_LOAD,
             "NETWORK::ERROR::LOAD: "
             "Invalid file format!");
    }

    size_t arr[4];
    read(fd, &arr, sizeof(size_t) * 4);

    network_t* net = net_init(arr[0], arr[1], arr[2], arr[3], 0, 0.f);

    for (size_t l = 0; l < net->L; l++)
    {
        size_t w_size = net->hidden_size * net->hidden_size;
        size_t b_size = net->hidden_size;
        
        if (l == 0)
            w_size = net->input_size*net->hidden_size;
        
        else if (l == net->L - 1)
        {
            w_size = net->output_size * net->hidden_size;
            b_size = net->output_size;
        }
        
        read(fd, net->b[l]->array, sizeof(double) * b_size);
        read(fd, net->w[l]->array, sizeof(double) * w_size);
    }

    close(fd);

    return net;
}

/**
 * @brief Save network's weights and biases in a file
 * 
 * @param net Network to save
 * @param dst File to save network to
 */
void net_save(network_t* net, const char* dst)
{
    FILE* fp = fopen(dst, "w+");

    unsigned int signature = 0xDEADBEEF;
    fwrite(&signature, sizeof(int), 1, fp);

    size_t arr[4] = { net->L, net->input_size,
                      net->hidden_size, net->output_size };

    fwrite(&arr, sizeof(size_t), 4, fp);

    for (size_t l = 0; l < net->L; l++)
    {
        size_t w_size = net->hidden_size * net->hidden_size;
        size_t b_size = net->hidden_size;
        
        if (l == 0)
            w_size = net->input_size*net->hidden_size;
        
        else if (l == net->L - 1)
        {
            w_size = net->output_size * net->hidden_size;
            b_size = net->output_size;
        }
        
        fwrite(net->b[l]->array, sizeof(double), b_size, fp);
        fwrite(net->w[l]->array, sizeof(double), w_size, fp);
    }

    fclose(fp);
}

/**
 * @brief Displays the network's parameters
 * 
 * @param net Neural network struct
 */
void net_summary(network_t* net)
{
    printf("Layers (Hidden):\t%zu\n", net->L);
    printf("Input size:\t\t%zu\n", net->input_size);
    printf("Hidden size:\t\t%zu\n", net->hidden_size);
    printf("Output size:\t\t%zu\n\n", net->output_size);
}

/**
 * @brief Train the network
 * 
 * @param net Neural network struct
 * @param epochs Amount of times the network should iterate on training
 */
void net_train(network_t* net, dataset_t* data, size_t epochs)
{
    printf("\n[TRAINING]\n\n");

    for (size_t e = 0; e < epochs; e++)
    {
        printf("Epoch %zu / %zu\n", e+1, epochs);
        
        data_shuffle(data);
        
        // Todo: Parallelize batch training
        for (size_t b = 0; b < data->n - net->batch_size + 1; b++)
        {
            for (size_t i = b; i < net->batch_size + b; i++)
            {
                _net_init_X(net, data->X[i]);
                _net_init_y(net, data->y[i]);

                _net_feed_forward(net);
                _net_backprop(net);
                _net_mini_batch_gradient_descent(net);
            }

            _net_update(net);
        }    
    }
    
    printf("\nCompleted %zu epochs!\n\n", epochs);
}

/**
 * @brief Calculate network accuracy on test dataset.
 *        Output is thresholded at 0.95.
 * 
 * @param net Neural network struct
 * @param dataset Test dataset
 */
void net_evaluate(network_t* net, dataset_t* dataset)
{
    double accuracy = 0.f;
    printf("\n[EVALUATING]\n");
    
    for (size_t p = 0; p < dataset->n; p++)
    {
        _net_init_X(net, dataset->X[p]);
        _net_init_y(net, dataset->y[p]);
        
        _net_feed_forward(net);
        _net_binarize_output(net, 0.95f);

        accuracy += _net_evaluate_prediction(net);
    }

    accuracy /= dataset->n;

    printf("\nNetwork accuracy: [%f%%]\n\n", accuracy * 100);
}

/**
 * @brief Predict output on network with single input
 * 
 * @param net Network to perform the prediction
 * @param X Input to predict with
 */
void net_predict(network_t* net, double* X)
{
    _net_init_X(net, X);
    _net_feed_forward(net);
        
    printf("[PREDICTED OUTPUT]:\n\n");
    for (size_t i = 0; i < net->output_size; i++)
        printf("\t%f\n", net->a[net->L-1]->array[i]);

    printf("\n");
}



/* ==== NETWORK INTERNAL API ==== */



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
    net->grad_w = calloc(net->L, sizeof(matrix_t*));
    net->grad_b = calloc(net->L, sizeof(matrix_t*));

    net->X = m_init(1, net->input_size);
    net->y = m_init(1, net->output_size);

    for (size_t l = 0; l < net->L - 1; l++)
    {
        net->a[l] = m_init(1, net->hidden_size);
        net->z[l] = m_init(1, net->hidden_size);
        net->b[l] = m_init(1, net->hidden_size);
        net->delta[l] = m_init(1, net->hidden_size);
        net->grad_b[l] = m_init(1, net->hidden_size);
    }

    net->w[0] = m_init(net->input_size, net->hidden_size);
    net->grad_w[0] = m_init(net->input_size, net->hidden_size);
    
    for (size_t l = 1; l < net->L - 1; l++)
    {
        net->grad_w[l] = m_init(net->hidden_size, net->hidden_size);
        net->w[l] = m_init(net->hidden_size, net->hidden_size);
    }

    net->a[net->L - 1] = m_init(1, net->output_size);
    net->z[net->L - 1] = m_init(1, net->output_size);
    net->b[net->L - 1] = m_init(1, net->output_size);
    net->delta[net->L - 1] = m_init(1, net->output_size);
    net->grad_w[net->L - 1] = m_init(net->hidden_size, net->output_size);
    net->grad_b[net->L - 1] = m_init(1, net->output_size);
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
        m_free(net->grad_b[l]);
        m_free(net->grad_w[l]);
    }
    
    free(net->a);
    free(net->z);
    free(net->w);
    free(net->b);
    free(net->delta);
    free(net->grad_b);
    free(net->grad_w);
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
 * @brief     Computes gradient descent
 *            on training examples on determined batch size 
 * 
 * @param net Neural network struct
 */
static void _net_mini_batch_gradient_descent(network_t* net)
{
    matrix_t* grad_w;
    matrix_t* a_T;
    
    for (int l = net->L - 1; l >= 0; l--)
    {
        if (l == 0)
            a_T = m_transpose(net->X);
        else
            a_T = m_transpose(net->a[l - 1]);

        grad_w = m_copy(net->grad_w[l]);
        
        m_mul(a_T, net->delta[l], grad_w);
        m_add(grad_w, net->grad_w[l], net->grad_w[l]);
        m_add(net->delta[l], net->grad_b[l], net->grad_b[l]);

        m_free(a_T);
        m_free(grad_w);
    }
}

/**
 * @brief Update network weights and biases with delta layer
 * 
 * @param net Neural network struct
 */
static void _net_update(network_t* net)
{
    double lr = net->lr / net->batch_size;

    for (int l = net->L - 1; l >= 0; l--)
    {
        // Weight update
        m_scalar_mul(net->grad_w[l], lr, net->grad_w[l]);
        m_sub(net->w[l], net->grad_w[l], net->w[l]);
        
        // Bias update
        m_scalar_mul(net->grad_b[l], lr, net->grad_b[l]);
        m_sub(net->b[l], net->grad_b[l], net->b[l]);
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

/**
 * @brief  Checks for equality between
 *         network prediction, and expected output
 * 
 * @param  net Neural network struct
 * @return int 1. if equal, 0. if not.
 */
static double _net_evaluate_prediction(network_t* net)
{
    double pred = 1.f;
    
    for(size_t i = 0; i < net->output_size; i++)
    {
        if (net->a[net->L - 1]->array[i] != net->y->array[i])
        {
            pred = 0.f;
            break;
        }
    }

    return pred;
}

/**
 * @brief Binarizes the network's prediction, for comparability with
 *        the expected output array. Threshold usually at 0.95.
 * 
 * @param net Neural network struct
 * @param threshold All values above the threshold are set to 1.
 *                  All values under the threshold are set to 0.
 */
static void _net_binarize_output(network_t* net, double threshold)
{
    for(size_t i = 0; i < net->output_size; i++)
    {
        if (net->a[net->L - 1]->array[i] > threshold)
            net->a[net->L - 1]->array[i] = 1.f;
        else
            net->a[net->L - 1]->array[i] = 0.f;
    }
}
