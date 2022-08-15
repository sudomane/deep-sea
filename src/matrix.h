/**
 * @file    matrix.h
 * @author  Philippe Bouchet (philippe.bouchet@epita.fr)
 * @brief   Pseudo-2D matrix implementation, with memory based operations.
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef MATRIX_H
#define MATRIX_H

#define MATRIX_FAILED_INITIALIZE        -1
#define MATRIX_OUT_OF_BOUNDS            -2
#define MATRIX_FAILED_MULTIPLICATION    -3
#define MATRIX_FAILED_ADDITION          -4
#define MATRIX_FAILED_SUBSTRACTION      -5
#define MATRIX_FAILED_HADAMARD          -6
#define MATRIX_FAILED_APPLY             -7

typedef unsigned long size_t;

typedef struct
__attribute__((packed, aligned(1)))
{
    double* array;
    
    size_t  size;
    size_t  n_row;
    size_t  n_col;
} matrix_t;

matrix_t*   m_init(size_t n_row, size_t n_col);
void        m_free(matrix_t* m);

void        m_set(matrix_t* m, size_t row, size_t col, double val);
double      m_get(matrix_t* m, size_t row, size_t col);
void        m_display(matrix_t* m);

matrix_t*   m_copy(matrix_t* m);
void        m_reset(matrix_t* m);
void        m_fill(matrix_t* m, double (*fun)(void));

void        m_mul(matrix_t* m1, matrix_t* m2, matrix_t* dst);
void        m_add(matrix_t* m1, matrix_t* m2, matrix_t* dst);
void        m_sub(matrix_t* m1, matrix_t* m2, matrix_t* dst);
void        m_hadamard(matrix_t* m1, matrix_t* m2, matrix_t* dst);
void        m_scalar_mul(matrix_t* m, double lambda, matrix_t* dst);
void        m_scalar_add(matrix_t* m, double lambda, matrix_t* dst);

matrix_t*   m_transpose(matrix_t* m);

void        m_apply_dst(matrix_t* m, double (*fun) (double), matrix_t* dst);
matrix_t*   m_apply(matrix_t* m, double (*fun)(double));


#endif // MATRIX_H