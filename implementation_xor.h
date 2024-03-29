// xor nn, 2024 M64. Schallner
#ifndef __XOR_H__
#define __XOR_H__
#include "nn_neuralnet.h"
#include <stdarg.h>

void colorprintf(int intensity, const char *f, ...);

void xor_visualizer(int size, NeuralNetwork *n);

double xor_fitness(NeuralNetwork *n);

#endif
