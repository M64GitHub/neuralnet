#include "evolution_ga.h"
#include <stdlib.h>

Individual *
initializeIndividual(int n_i_neurons, int n_o_neurons, int n_hidden_layers,
                     int n_neurons_per_hlayer,
                     NN_Activation_Function_ID activation_function_type) {
  Individual *individual;
  individual = (Individual *)malloc(sizeof(Individual));

  individual->age = 0;
  individual->fitness = 1;
  individual->network =
      initializeNetwork(n_i_neurons, n_o_neurons, n_hidden_layers,
                        n_neurons_per_hlayer, activation_function_type);

  return individual;
}

void freeIndividual(Individual *I) {}

Population *initializePopulation(int pop_size) { return 0; }

void freePopulation(Population *P) {}

World *initializeWorld(int pop_size, double mut_rate_ind, double mut_rate,
                       double mut_amnt, double sel_pressure,
                       double crossovr_rt) {
  return 0;
}

void freeWorld(World *W) {}
