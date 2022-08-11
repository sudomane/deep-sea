#ifndef MATRIX_H
#define MATRIX_H

/* Neural network internal API */

typedef unsigned long size_t;

typedef struct
{
    double* array;
    
    size_t size; // Total size of array n_row * n_col
    size_t n_row;
    size_t n_col;
} matrix_t;

matrix_t* m_init(size_t n_row, size_t n_col);
void m_free(matrix_t* m);

void m_set(matrix_t* m, size_t row, size_t col, double val);
double m_get(matrix_t* m, size_t row, size_t col);
void m_display(matrix_t* m);

matrix_t* m_copy(matrix_t* m);
void m_fill(matrix_t* m, double (*fun)(void));

void m_mul(matrix_t* m1, matrix_t* m2, matrix_t* dst);
void m_add(matrix_t* m1, matrix_t* m2, matrix_t* dst);
void m_sub(matrix_t* m1, matrix_t* m2, matrix_t* dst);
void m_hadamard(matrix_t* m1, matrix_t* m2, matrix_t* dst);
void m_scalar_mul(matrix_t* m, double lambda, matrix_t* dst);
void m_scalar_add(matrix_t* m, double lambda, matrix_t* dst);

matrix_t* m_transpose(matrix_t* m);

void m_apply_dst(matrix_t* m, double (*fun) (double), matrix_t* dst);
matrix_t* m_apply(matrix_t* m, double (*fun)(double));


#endif // MATRIX_H