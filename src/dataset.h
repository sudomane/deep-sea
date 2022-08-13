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

typedef struct 
{
    double** X;
    double** y;
}dataset_t;




#endif //DATASET_H