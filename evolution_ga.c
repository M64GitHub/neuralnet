#include "evolution_ga.h"
#include <stdlib.h>

Individual *NN_Individual_initialize(NeuralNetwork *reference_network) {
  Individual *individual;
  individual = (Individual *)malloc(sizeof(Individual));

  individual->age = 0;
  individual->fitness = 0.0;
  individual->network = NN_Network_initialize(
      reference_network->num_inputs, reference_network->num_outputs,
      reference_network->num_h_layers, reference_network->neurons_per_h_layer,
      reference_network->activation_function_id);
  return individual;
}

void NN_Individual_free(Individual *I) {
  if (I->network)
    NN_Network_free(I->network);
  free(I);
}

// Creates and returns initialized Population
Population *NN_Population_initialize(int pop_size, NeuralNetwork *ref_nw) {
  Population *p = (Population *)malloc((sizeof(Population)));

  p->size = pop_size;
  p->individuals = (Individual **)malloc((sizeof(Individual *)) * pop_size);
  p->age = 0;
  p->current_best_fitness_val = 0.0;
  p->current_best_individuals = 0;
  p->current_best_individuals_size = 0;
  p->reference_network = ref_nw;

  // create nn clones, and individuals
  for (int i = 0; i < pop_size; i++) {
    p->individuals[i] = NN_Individual_initialize(ref_nw);
  }

  return p;
}

void NN_Population_free(Population *P) {
  for (int i = 0; i < P->size; i++)
    NN_Individual_free(P->individuals[i]);
  free(P->individuals);
  // don't free reference nw yet, we free, where we malloc
  free(P);
}

// Creates and returns initialized World
World *NN_World_initialize(int pop_size, double mut_rate_ind, double mut_rate,
                          double mut_amnt, double sel_pressure,
                          double crossovr_rt, int num_populations,
                          NeuralNetwork *ref_nw) {
  World *w = (World *)malloc((sizeof(World)));

  w->population_size = pop_size;
  w->mutation_rate = mut_rate;
  w->mutation_rate_individuals = mut_rate_ind;
  w->mutation_amount = mut_amnt;
  w->selection_pressure = sel_pressure;
  w->crossover_rate = crossovr_rt;
  w->num_populations = num_populations;
  // Create Populations
  w->populations =
      (Population **)malloc((sizeof(Population *)) * num_populations);

  for (int i = 0; i < num_populations; i++)
    w->populations[i] = NN_Population_initialize(pop_size, ref_nw);

  w->reference_network = ref_nw;

  return w;
}

void NN_World_free(World *W) {
  if (!W)
    return;

  for (int i = 0; i < W->num_populations; i++)
    NN_Population_free((W->populations[i]));
  free(W->populations);

  if (W->reference_network)
    NN_Network_free(W->reference_network);

  free(W);
}

void NN_World_fill_rand(World *w) {
}
