/**
 * @file    dataset.c
 * @author  Philippe Bouchet (philippe.bouchet@epita.fr)
 * @brief   Dataset API implementation.
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "dataset.h"

#include <err.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "utils.h"

/* Internal API forward declaration. */

static int _data_reverse_int(int i);


/* ==== DATASET PUBLIC API ==== */

/**
 * @brief Initialize dataset
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

/**
 * @brief Load an MNIST dataset into the dataset struct
 * 
 * @param path Path to MNIST image or label dataset 
 * @param data Dataset struct to copy MNIST dataset to
 * @param load_type Dataset to load (images, or labels).
 *                  LOAD_IMAGES to load images, LOAD_LABELS to load labels
 */
void data_load_mnist(const char* path, dataset_t* data, int load_type)
{
    printf("\n[LOADING DATASET]\n\n");
    int fd = open(path, O_RDONLY);
    
    if (fd == -1)
    {
        errx(DATASET_FAILED_LOAD,
             "DATASET::ERROR::LOAD: "
             "Could not open file: %s (Invalid path, or corrupted file).",
             path);
    }

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

        if (data->n_input != (size_t) n_rows * (size_t) n_cols)
        {
            errx(DATASET_INPUT_MISMATCH,
                 "DATASET::ERROR::INCOMPATIBLE SIZE: "
                 "Input size mismatch (%zu) with dataset input (%zu)",
                 data->n_input, (size_t) (n_rows*n_cols));
        }
    }

    if (load_type == LOAD_IMAGES)
        printf("Found:\t%d\t[Images] of (%dx%d)\n", n_images, n_rows, n_cols);
    else
        printf("Found:\t%d\t[Labels]\n", n_images);
  

    if (data->n > (size_t) n_images)
    {
        warnx("DATASET::WARNING::INCOMPATIBLE SIZE: "
              "Expected %zu elements, got %zu. Lowering to %zu.",
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
                
                double normalized = normalize((double) tmp);
                data->X[n][i] = normalized;
            }
        }
    }
    
    else
    {
        for (size_t n = 0; n < data->n; n++)
        {
            unsigned char tmp;
            read(fd, &tmp, sizeof(unsigned char));

            data->y[n][(int)tmp] = 1;
        }
    }

    if (load_type == LOAD_IMAGES)
        printf("Loaded: %zu\t[Images] of (%dx%d)\n\n",
                data->n, n_rows, n_cols);
    else
        printf("Loaded: %zu\t[Labels]\n\n", data->n);

    close(fd);
}


/* ==== DATASET INTERNAL API ==== */


/**
 * @brief  Reverse byte order in integer i
 * 
 * @param  i Integer to reverse
 * @return Reversed integer
 */
static int _data_reverse_int(int i)
{
    unsigned char c_1, c_2, c_3, c_4;
    
    c_1 = i & 255;
    c_2 = (i >> 8) & 255;
    c_3 = (i >> 16) & 255;
    c_4 = (i >> 24) & 255;

    return ((int) c_1 << 24) + ((int) c_2 << 16) + ((int) c_3 << 8) + c_4;
}