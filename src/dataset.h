/**
 * @file    dataset.h
 * @author  Philippe Bouchet (philippe.bouchet@epita.fr)
 * @brief   Dataset API for easy user interaction with the neural network
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef DATASET_H
#define DATASET_H

#define LOAD_IMAGES 0
#define LOAD_LABELS 1

#define DATASET_FAILED_LOAD     -1
#define DATASET_INPUT_MISMATCH  -2

typedef unsigned long size_t;

typedef struct
__attribute__((packed, aligned(1))) 
{
    size_t      n;           // Number of elements in dataset
    size_t      n_input;     // Input size
    size_t      n_output;    // Output size
    double**    X;
    double**    y;
}dataset_t;

dataset_t*  data_init(size_t n, size_t input_size, size_t output_size);
void        data_free(dataset_t* data);

void        data_display(dataset_t* data);
void        data_shuffle(dataset_t* data);

void        data_load_mnist(const char* path, dataset_t* data, int load_type);


#endif //DATASET_H