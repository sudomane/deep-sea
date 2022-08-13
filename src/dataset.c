/**
 * @file    dataset.c
 * @author  Philippe Bouchet (philippe.bouchet@epita.fr)
 * @brief 
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "dataset.h"

#include <err.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

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

static int _data_reverse_int(int i)
{
    unsigned char c_1, c_2, c_3, c_4;
    
    c_1 = i & 255;
    c_2 = (i >> 8) & 255;
    c_3 = (i >> 16) & 255;
    c_4 = (i >> 24) & 255;

    return ((int) c_1 << 24) + ((int) c_2 << 16) + ((int) c_3 << 8) + c_4;
}

void data_load_mnist(const char* path, dataset_t* data, int load_type)
{
    int fd = open(path, O_RDONLY);
    
    if (fd == -1)
        errx(-1, "ERROR::LOAD DATASET: Could not open file: %s "
                 "(Invalid path, or corrupted file).", path);

    int magic =     0;
    int n_images =  0;
    int n_rows =    0;
    int n_cols =    0;

    read(fd, &magic, sizeof(int));
    read(fd, &n_images, sizeof(int));

    if (load_type == LOAD_IMAGES)
    {
        read(fd, &n_rows, sizeof(int));
        read(fd, &n_cols, sizeof(int));
    }

    magic = _data_reverse_int(magic);
    n_images = _data_reverse_int(n_images);

    if (load_type == LOAD_IMAGES)
    {
        n_rows = _data_reverse_int(n_rows);
        n_cols = _data_reverse_int(n_cols);
    }

    if (load_type == LOAD_IMAGES)
        printf("READ: %d %d %d\n", n_images, n_rows, n_cols);
    else
        printf("READ: %d \n", n_images);
  

    if (data->n >= (size_t) n_images)
    {
        warnx("WARNING::LOAD DATASET: Expected %zu elements, got %zu. "
              "Setting data->n to %zu.",
              (size_t) n_images, data->n, (size_t) n_images);
        data->n = n_images;
    }

    if (load_type == LOAD_IMAGES)
    {
        for (size_t n = 0; n < data->n; n++)
        {
            for (size_t i = 0; i < data->n_input; i++)
            {
                unsigned char tmp;
                read(fd, &tmp, sizeof(unsigned char));

                data->X[n][i] = (double) tmp / 255.f;
            }
        }
    }
    
    else
    {
        for (size_t n = 0; n < data->n; n++)
        {
            for(size_t i = 0; i < data->n_output; i++)
            {
                unsigned char tmp;
                read(fd, &tmp, sizeof(unsigned char));

                data->y[n][i]= (int)tmp;
            }
        }
    }

    close(fd);
}