#include "evolution_ga.h"
#include <stdlib.h>

Individual *
initializeIndividual(NeuralNetwork *network) {
  Individual *individual;
  individual = (Individual *)malloc(sizeof(Individual));

  individual->age = 0;
  individual->fitness = 1;
  individual->network = network;

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

// --

void mutate(){};

void runPopulation(){};
void evolutePopulation(){}


