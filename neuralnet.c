// artifical neural network for test and learning purposes. 2024, M64 Schallner
// <mario.a.schallner@gmail.com>
// A forward propagation artifical neural network (ANN) very close to a
// multi-layer perceptron.
// Evolution-theory and genetic algorithm based learning.
// Various activation functions to choose from.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// -- CONSTANTS for learning
// Learning is based on evolution theory.
#define NUM_POPULATIONS (1)
#define POPULATION_SIZE (10)
#define PCT_RECOMBINATORS (25)
// ... and genetic algorithms

// -- structs

// Struct for neuron configuration
// We make the convention that a neuron always has 1 input more than specified.
// We choose input 0 for this, and set it's value to 1 on purpose. This way we
// we further on might have advantages in calculation as we only need to apply
// the same kind of operation to each input to get the weighted sum, with an 
// added bias that will be represented by weight 0.
// (And we can work with operating on weights only).
// 1 * x = x, so it's like adding x to the weighted sum, and we have our bias.
typedef struct S_Neuron {
  int num_inputs;     // number of inputs
  double *weights;    // weights[0] is bias         | Both arrays will have
  double *input_vals; // (when input_val[0] = 1)    | the size: num_inputs+1
                      // We need not store them here, because these values are
                      // read from the outputs of the prev layer's neurons. But
                      // we do anyways, for easier status display later maybe.
                      // -> *input_vals must look like: { 1.0, i1, i2, ... in }
  double output;      // the neuron's output value
} Neuron;

// Struct for the neural network
typedef struct S_NeuralNetwork {
  int num_inputs;  // number of input neurons
  int num_outputs; // number of output neurons (ie tags, ...)

  int num_h_layers;        // number of hidden layers
  int neurons_per_h_layer; // neurons per hidden layer

  Neuron *i_layer;   // input layer, 1D array of neurons
  Neuron *o_layer;   // output layer, 1D array of neurons
  Neuron **h_layers; // the hidden layers 2D array of neurons, (as we
                     // use no explicit layer struct/type anyways)
} NeuralNetwork;

// -- helpers

// Sigmoid activation function
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

// -- nn functions

// Function to initialize a neural network
NeuralNetwork *initializeNetwork(int n_i_neurons, int n_o_neurons,
                                 int n_hidden_layers,
                                 int n_neurons_per_hlayer) {
  NeuralNetwork *network = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
  // Allocate memory for neurons / layers
  network->i_layer = (Neuron *)malloc(sizeof(Neuron) * n_i_neurons);
  network->o_layer = (Neuron *)malloc(sizeof(Neuron) * n_o_neurons);
  network->h_layers = (Neuron **)malloc(sizeof(Neuron) * n_hidden_layers *
                                        n_neurons_per_hlayer);

  // Init neurons (set bias input to 1, all else to 0)

  // input layer: as many neurons as inputs to the network
  for (int i = 0; i < n_i_neurons; i++) {
    network->i_layer[i].num_inputs = 1;
    // input neurons have 1 input (+1 as bias)
    network->i_layer[i].input_vals = (double *)malloc(sizeof(double) * (1 + 1));
    network->i_layer[i].weights = (double *)malloc(sizeof(double) * (1 + 1));

    // init neurons weights, values to 0
    network->i_layer[i].input_vals[0] = 1.0; // fixed bias multiplicator
    network->i_layer[i].weights[0] = 0.0;    // bias

    network->i_layer[i].input_vals[1] = 0.0; // input neuron input = 0
    network->i_layer[i].weights[1] = 0.0;    // input neuron weight = 0

    network->i_layer[i].output = 0.0; // input neuron output = 0
  }                                   // i, input layer neuron number

  // output layer: as many neurons as specified as network outputs
  for (int o = 0; o < n_o_neurons; o++) {
    network->o_layer[o].num_inputs = n_neurons_per_hlayer + 1;
    // input neurons have as many inputs as hidden layer neurons (+1 as bias)
    network->o_layer[o].input_vals =
        (double *)malloc(sizeof(double) * (n_neurons_per_hlayer + 1));

    // init neuron's bias to 0
    network->i_layer[o].input_vals[0] = 1.0; // fixed bias multiplicator 1
    network->i_layer[o].weights[0] = 0.0;    // bias = 0

    // init neuron's weights, values to 0
    for (int i = 1; i < (n_neurons_per_hlayer + 1); i++) {
      network->i_layer[o].input_vals[i] = 0.0; // input neuron input = 0
      network->i_layer[o].weights[i] = 0.0;    // input neuron weight = 0
    }

    network->i_layer[o].output = 0.0; // input neuron output = 0
  }                                   // o, output layer neuron number

  return network;
}

// Function to free memory allocated for the neural network
void freeNetwork(NeuralNetwork *network) {
}

// Function to set input values for the input layer
void setInputValues(NeuralNetwork *network, double *inputValues) {
}

// Function for forward propagation to calculate the output of the network
void forwardPropagation(NeuralNetwork *network) {
}

int main() {
  return 0;
}