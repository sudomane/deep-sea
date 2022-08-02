#ifndef MATRIX_H
#define MATRIX_H

typedef unsigned long size_t;

typedef struct
{
    double* array;
    
    size_t row;
    size_t col;
    
    size_t size;
} matrix_2D_t;

matrix_2D_t* init_matrix(size_t height, size_t width);
void free_matrix(matrix_2D_t* mat);

void display_matrix(matrix_2D_t* mat);

void set_at(matrix_2D_t* mat, int x, int y, double val);
double get_at(matrix_2D_t* mat, int x, int y);

void reset_matrix(matrix_2D_t* mat);
void fill_matrix(matrix_2D_t* mat, double val);


#endif // MATRIX_H