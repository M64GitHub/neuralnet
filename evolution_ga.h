#ifndef __EVOLUTION_GA_H__
#define __EVOLUTION_GA_H__

#include "neuralnet.h"

// Hyper parameters, specifying the environment
typedef struct S_World {
  int population_size;
  double mutation_rate;
  double selection_pressure;
  double crossover_rate;
} World;

typedef struct S_Individual {
  NeuralNetwork *network;
  double fitness;
} Individual;

typedef struct S_Population {
  int size;
  Individual *individuals;

  double current_best_fitness_val;
  Individual **current_best_individuals; // array of pointers
  int current_best_individuals_size;
} Population;

#endif
