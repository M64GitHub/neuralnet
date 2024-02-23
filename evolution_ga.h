#ifndef __EVOLUTION_GA_H__
#define __EVOLUTION_GA_H__

#include "neuralnet.h"

// Hyper parameters, specifying the environment
typedef struct S_World {
  int population_size;
  double mutation_rate_individuals; // how many individuals will mutate
  double mutation_rate;             // how many weights will mutate
  double mutation_strength;         // how much a weight can mutate
  double selection_pressure; // how many parents need to be chosen from the best
                             // (fittest) individuals
  double crossover_rate;     // how many individuals will be selected as parents
} World;

typedef struct S_Individual {
  NeuralNetwork *network;
  double fitness;
  int age;
} Individual;

typedef struct S_Population {
  int size;
  Individual *individuals;

  double current_best_fitness_val;
  Individual **current_best_individuals; // array of pointers
  int current_best_individuals_size;
  int age;
} Population;

#endif
