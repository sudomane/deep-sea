#ifndef UTILS_H
#define UTILS_H

/* Neural network internal API */

double normalized_rand(void);
double sigmoid(double x);
double d_sigmoid(double x);
double relu(double x);
double d_relu(double x);

#endif // UTILS_H