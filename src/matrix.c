/**
 * @file    matrix.c
 * @author  Philippe Bouchet (philippe.bouchet@epita.fr)
 * @brief   Matrix implementation.
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "matrix.h"

#include <err.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

matrix_t* m_init(size_t n_row, size_t n_col)
{
    matrix_t* m = malloc(sizeof(matrix_t));

    if (m == NULL)
        errx(MATRIX_FAILED_INITIALIZE,
            "Not enough memory to initialize matrix!");

    m->array = calloc(n_row * n_col, sizeof(double));

    if (m->array == NULL)
        errx(MATRIX_FAILED_INITIALIZE,
            "Not enough memory to initialize matrix array!");

    m->n_row = n_row;
    m->n_col = n_col;
    m->size = n_row * n_col;

    return m;
}

void m_free(matrix_t* m)
{   
    free(m->array);
    free(m);
}

void m_display(matrix_t* m)
{
    printf("\t(%zu, %zu):\n", m->n_row, m->n_col);
    for (size_t i = 0; i < m->n_row; i++)
    {
        printf("\t\t");
        for (size_t j = 0; j < m->n_col; j++)
        {
            double val = m_get(m, i, j);
            printf("%f ", val);
        }

        printf("\n");
    }
}

matrix_t* m_copy(matrix_t* m)
{
    matrix_t* m_ = m_init(m->n_row, m->n_col);
    
    for (size_t i = 0; i < m->size; i++)
        m_->array[i] = m->array[i];

    return m_;
}

void m_fill(matrix_t* m, double (*fun)(void))
{
    for (size_t i = 0; i < m->size; i++)
        m->array[i] = fun();
}

double m_get(matrix_t* m, size_t row, size_t col)
{
    if (row >= m->n_row || col >= m->n_col)
        errx(MATRIX_OUT_OF_BOUNDS,
            "GET: Values (%zu, %zu) out of bounds for matrix (%zu, %zu)",
            row, col, m->n_row, m->n_col);

    return m->array[m->n_row * col + row];
}

void m_set(matrix_t* m, size_t row, size_t col, double val)
{
    if (row >= m->n_row || col >= m->n_col)
        errx(MATRIX_OUT_OF_BOUNDS,
            "SET: Values (%zu, %zu) out of bounds for matrix (%zu, %zu)",
            row, col, m->n_row, m->n_col);

    m->array[m->n_row * col + row] = val;
}

void m_mul(matrix_t* m1, matrix_t* m2, matrix_t* dst)
{
    if (m1->n_col != m2->n_row)
        errx(MATRIX_FAILED_MULTIPLICATION,
            "MUL: Incompatible shapes (%zu, %zu) and (%zu, %zu)",
            m1->n_row, m1->n_col, m2->n_row, m2->n_col);

    if (m1->n_row != dst->n_row || m2->n_col != dst->n_col)
        errx(MATRIX_FAILED_MULTIPLICATION,
            "MUL: Incompatible dst. Got (%zu, %zu), expected (%zu, %zu)",
            m1->n_row, m2->n_col, dst->n_row, dst->n_col);
        
    for (size_t i = 0; i < m1->n_row; i++)
    {
        for (size_t j = 0; j < m2->n_col; j++)
        {
            double val = 0.f;
            
            for (size_t k = 0; k < m1->n_col; k++)
            {
                double m1_val = m_get(m1, i, k);
                double m2_val = m_get(m2, k, j);
                val += m1_val * m2_val;
            }

            m_set(dst, i, j, val);
        }
    }
}

void m_add(matrix_t* m1, matrix_t* m2, matrix_t* dst)
{
    if (m1->n_col != m2->n_col || m1->n_row != m2->n_row)
        errx(MATRIX_FAILED_ADDITION,
            "ADD: Incompatible shapes (%zu, %zu) and (%zu, %zu)",
            m1->n_row, m1->n_col, m2->n_row, m2->n_col);

    if (m1->n_col != dst->n_col || m1->n_row != dst->n_row)
        errx(MATRIX_FAILED_ADDITION,
            "ADD: Incompatible dst. Got (%zu, %zu), expected (%zu, %zu)",
            m1->n_row, m1->n_col, dst->n_row, dst->n_col);

    for (size_t i = 0; i < m1->size; i++)
        dst->array[i] = m1->array[i] + m2->array[i];
}

void m_sub(matrix_t* m1, matrix_t* m2, matrix_t* dst)
{
    if (m1->n_col != m2->n_col || m1->n_row != m2->n_row)
        errx(MATRIX_FAILED_SUBSTRACTION,
            "SUB: Incompatible shapes (%zu, %zu) and (%zu, %zu)",
            m1->n_row, m1->n_col, m2->n_row, m2->n_col);

    if (m1->n_col != dst->n_col || m1->n_row != dst->n_row)
        errx(MATRIX_FAILED_SUBSTRACTION,
            "SUB: Incompatible dst. Got (%zu, %zu), expected (%zu, %zu)",
            m1->n_row, m1->n_col, dst->n_row, dst->n_col);

    for (size_t i = 0; i < m1->size; i++)
        dst->array[i] = m1->array[i] - m2->array[i];
}

void m_scalar_mul(matrix_t* m, double lambda, matrix_t* dst)
{
    for (size_t i = 0; i < m->size; i++)
        dst->array[i] = m->array[i] * lambda;
}

void m_scalar_add(matrix_t* m, double lambda, matrix_t* dst)
{
    for (size_t i = 0; i < m->size; i++)
        dst->array[i] = m->array[i] + lambda;
}

void m_hadamard(matrix_t* m1, matrix_t* m2, matrix_t* dst)
{
    if (m1->n_col != m2->n_col || m1->n_row != m2->n_row)
        errx(MATRIX_FAILED_HADAMARD,
            "HADAMARD: Incompatible shapes (%zu, %zu) and (%zu, %zu).",
            m1->n_row, m1->n_col, m2->n_row, m2->n_col);
    
    if (m1->n_col != dst->n_col || m1->n_row != dst->n_row)
        errx(MATRIX_FAILED_HADAMARD,
            "HADAMARD: Incompatible dst. Got (%zu, %zu), expected (%zu, %zu)",
            m1->n_row, m1->n_col, dst->n_row, dst->n_col);
    
    for (size_t i = 0; i < m1->size; i++)
        dst->array[i] = m1->array[i] * m2->array[i];
}

matrix_t* m_transpose(matrix_t* m)
{
    matrix_t* m_t = m_init(m->n_col, m->n_row);

    for (size_t i = 0; i < m->n_row; i++)
    {
        for (size_t j = 0; j < m->n_col; j++)
        {
            double val = m_get(m, i, j);
            m_set(m_t, j, i, val);
        }
    }

    return m_t;
}

void m_apply_dst(matrix_t* m, double (*fun)(double), matrix_t* dst)
{
    if (m->n_row != dst->n_row || m->n_col != dst->n_col)
        errx(MATRIX_FAILED_APPLY,
            "APPLY: Incompatible dst.  Got (%zu, %zu), expected (%zu, %zu)",
            m->n_row, m->n_col, dst->n_row, dst->n_col);

    for (size_t i = 0; i < m->size; i++)
        dst->array[i] = fun(m->array[i]);
}

matrix_t* m_apply(matrix_t* m, double (*fun)(double))
{
    matrix_t* dst = m_init(m->n_row, m->n_col);
    
    for (size_t i = 0; i < m->size; i++)
        dst->array[i] = fun(m->array[i]);

    return dst;
}
