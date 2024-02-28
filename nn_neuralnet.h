#ifndef __NEURALNET_H__
#define __NEURALNET_H__
// artifical neural network for test and learning purposes. 2024, M64 Schallner
// <mario.a.schallner@gmail.com>
// A forward propagation artifical neural network (ANN).
// Evolution-theory and genetic algorithm based learning.

// -- type definitions

// -- enums

typedef enum {
  NN_AF_NONE,
  NN_AF_SIGMOID,
  NN_AF_RELU
} NN_Activation_Function_ID;

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
  NN_Activation_Function_ID af;
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
                     // [num_h_layers][neurons_per_h_layer]
  NN_Activation_Function_ID activation_function_id;
} NeuralNetwork;

// -- activation functions
double NN_af_sigmoid(double x);
double NN_af_relU(double x);

// -- neuron functions
double NN_Neuron_weightedsum(Neuron *n);
double NN_Neuron_process(Neuron *n);

// -- neural net functions
NeuralNetwork *
NN_Network_initialize(int n_i_neurons, int n_o_neurons, int n_hidden_layers,
                      int n_neurons_per_hlayer,
                      NN_Activation_Function_ID activation_function_type);
void NN_Network_free(NeuralNetwork *network);
void NN_Network_input_values_set(NeuralNetwork *network, double *inputValues);
void NN_Network_propagate_forward(NeuralNetwork *network);
void NN_Network_dump(NeuralNetwork *network);

#endif
