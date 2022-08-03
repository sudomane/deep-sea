#include "matrix.h"

#include <err.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// row = i
// col = j
matrix_2D_t* init_matrix(size_t row, size_t col)
{
    matrix_2D_t* mat = malloc(sizeof(matrix_2D_t));

    if (mat == NULL)
    {
        errx(-1, "Not enough memory to initialize matrix!\n");
    }

    mat->array = calloc(row * col, sizeof(double));

    if (mat->array == NULL)
    {
        errx(-1, "Not enough memory to initialize matrix array!\n");
    }

    mat->row = row;
    mat->col = col;
    mat->size = row * col;

    return mat;
}

void free_matrix(matrix_2D_t* mat)
{   
    free(mat->array);
    free(mat);
}

void reset_matrix(matrix_2D_t* mat)
{
    // TODO: Fix!
    //memset(mat->array, 0, mat->size);
    for (size_t i = 0; i < mat->size; i++)
        mat->array[i] = 0;
}

void fill_matrix(matrix_2D_t* mat, double val)
{
    // TODO: Fix!
    //memset(mat->array, val, mat->size * sizeof(double));
    for (size_t i = 0; i < mat->size; i++)
        mat->array[i] = val;

}

void display_matrix(matrix_2D_t* mat)
{
    for (size_t i = 0; i < mat->row; i++)
    {
        for (size_t j = 0; j < mat->col; j++)
        {
            double val = get_at(mat, i, j);

            if (val == -1)
                printf("\t ");
            else
                printf("%f ", get_at(mat, i, j));
        }

        printf("\n");
    }
}

double get_at(matrix_2D_t* self, int i, int j)
{
    if (i >= self->row || j >= self->col || i < 0 || j < 0)
        errx(-1, "Cannot get here, out of matrix bounds!\n");

    return self->array[self->row * j + i];
}

void set_at(matrix_2D_t* self, int i, int j, double val)
{
    if (i >= self->row || j >= self->col || i < 0 || j < 0)
        errx(-1, "Cannot set here, out of matrix bounds!\n");

    self->array[self->row * j + i] = val;
}

size_t matrix_size(matrix_2D_t* mat)
{
    size_t size = 0;

    for (size_t i = 0; i < mat->size; i++)
    {
        if (mat->array[i] == -1)
            continue;
        size++;
    }

    return size;
}