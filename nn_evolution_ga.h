#ifndef __EVOLUTION_GA_H__
#define __EVOLUTION_GA_H__

#include "nn_neuralnet.h"

typedef struct S_Individual {
  NeuralNetwork *network;
  double fitness;
  int age;
} Individual;

typedef struct S_Population {
  int size;
  Individual **individuals; // list of ptrs

  double current_best_fitness_val;
  Individual **current_best_individuals; // array of pointers
  int current_best_individuals_size;
  int age;
  NeuralNetwork *reference_network; // nn config for Individuals
  // and Population
} Population;

// Hyper parameters, specifying the environment
typedef struct S_World {
  int population_size;
  double mutation_rate_individuals; // how many individuals will mutate
  double mutation_rate;             // how many weights will mutate
  double mutation_amount;           // how much a weight can mutate
  double selection_pressure; // how many parents need to be chosen from the best
                             // (fittest) individuals
  double crossover_rate;     // how many individuals will be selected as parents
  Population **populations;  // list of ptrs
  int num_populations;

  NeuralNetwork *reference_network;
} World;

// --

Individual *NN_Individual_initialize(NeuralNetwork *network);
void NN_Individual_free(Individual *I);

// --

Population *NN_Population_initialize(int pop_size, NeuralNetwork *ref_nw);
void NN_Population_free(Population *P);
void NN_Population_list_individuals(Population *P);
void NN_Population_dump_individuals(Population *P);
void NN_Population_run_forward_propagation(Population *P);

// --

World *NN_World_initialize(
    int pop_size, double mut_rate_ind, double mut_rate, double mut_amnt,
    double sel_pressure, double crossovr_rt, int num_populations,
    NeuralNetwork
        *reference_network); // will be passed down to population to individual

void NN_World_free(World *W);

// --

void NN_World_fill_rand(World *w);

#endif
