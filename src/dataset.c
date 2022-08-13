/**
 * @file    dataset.c
 * @author  Philippe Bouchet (philippe.bouchet@epita.fr)
 * @brief 
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "dataset.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static void _data_init_XOR(dataset_t* data)
{
    data->X[0][0] = 0.f;
    data->X[0][1] = 0.f;
    data->y[0][0] = 0.f;

    data->X[1][0] = 0.f;
    data->X[1][1] = 1.f;
    data->y[1][0] = 1.f;

    data->X[2][0] = 1.f;
    data->X[2][1] = 0.f;
    data->y[2][0] = 1.f;
    
    data->X[3][0] = 1.f;
    data->X[3][1] = 1.f;
    data->y[3][0] = 0.f;
}

/**
 * @brief 
 * 
 * @param n Number of elements in dataset 
 * @param input_size
 * @param output_size 
 * @return dataset_t* 
 */
dataset_t* data_init(size_t n, size_t input_size, size_t output_size)
{
    dataset_t* data = malloc(sizeof(dataset_t));
    
    data->n = n;
    data->n_input = input_size;
    data->n_output = output_size;
    
    data->X = calloc(n, sizeof(double*));
    data->y = calloc(n, sizeof(double*));

    for (size_t i = 0; i < n; i++)
    {
        data->X[i] = calloc(input_size, sizeof(double));
        data->y[i] = calloc(output_size, sizeof(double));
    }

    // TEST XOR
    _data_init_XOR(data);
    
    return data;
}

/**
 * @brief Free's the dataset
 * 
 * @param data 
 */
void data_free(dataset_t* data)
{
    for (size_t i = 0; i < data->n; i++)
    {
        free(data->X[i]);
        free(data->y[i]);
    }

    free(data->X);
    free(data->y);

    free(data);
}

/**
 * @brief Prints the dataset to stdout
 * 
 * @param data Dataset to display
 */
void data_display(dataset_t* data)
{
    printf("X:\n");
    for (size_t n = 0; n < data->n; n++)
    {
        for (size_t i = 0; i < data->n_input; i++)
        {
            printf("%f ", data->X[n][i]);
        }

        printf("\n");
    }

    printf("\ny:\n");
    for (size_t n = 0; n < data->n; n++)
    {
        for (size_t i = 0; i < data->n_output; i++)
        {
            printf("%f ", data->y[n][i]);
        }

        printf("\n");
    }
}

/**
 * @brief Randomly shuffle the dataset
 * 
 * @param data Dataset to shuffle
 */
void data_shuffle(dataset_t* data)
{
    for (size_t i = 0; i < data->n; i++)
    {
        size_t r = i + rand() / (RAND_MAX / (data->n - i) + 1);

        double* X_tmp = data->X[r];
        double* y_tmp = data->y[r];

        data->X[r] = data->X[i];
        data->y[r] = data->y[i];

        data->X[i] = X_tmp;
        data->y[i] = y_tmp;
    }
}