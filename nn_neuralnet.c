// artifical neural network for test and learning purposes. 2024, M64 Schallner
// <mario.a.schallner@gmail.com>
// A forward propagation artifical neural network (ANN).
// Evolution-theory and genetic algorithm based learning.

#include "nn_neuralnet.h"
#include "nn_timing.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// -- activation functions

// Sigmoid activation function
double NN_af_sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

// relU activation function
double NN_af_relU(double x) { return (x <= 0.0) ? 0.0 : x; }

// -- neuron functions

double NN_Neuron_weightedsum(Neuron *n) {
  double ws = 0.0;
  for (int i = 0; i < n->num_inputs+1; i++) {
    ws += n->input_vals[i] * n->weights[i];
  }
  return ws;
}

double NN_Neuron_process(Neuron *n) {
  double ws = 0.0;
  double output = 0.0;

  // inlined for speed (will be done by compiler optimimization, too. anyways)
  for (int i = 0; i < n->num_inputs+1; i++) {
    ws += n->input_vals[i] * n->weights[i];
  }

  switch (n->af) {
  case NN_AF_NONE:
    output = ws;
    break;
  case NN_AF_SIGMOID:
    // output = sigmoid(ws);
    output = 1.0 / (1.0 + exp(-ws)); // sigmoid inlined for speed
    break;
  case NN_AF_RELU:
    // output = relU(ws);
    output = (ws <= 0.0) ? 0.0 : ws; // relU inlined for speed
    break;
  default:
    // output = relU(ws);
    output = (ws <= 0.0) ? 0.0 : ws; // relU inlined for speed
  }

  n->output = output;
  return output;
}

// -- nn functions

// Function to initialize a neural network
NeuralNetwork *
NN_Network_initialize(int n_i_neurons, int n_o_neurons, int n_hidden_layers,
                      int n_neurons_per_hlayer,
                      NN_Activation_Function_ID activation_function_type) {
  NeuralNetwork *network = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));

  // -- set nn layout specs
  network->num_inputs = n_i_neurons;
  network->num_outputs = n_o_neurons;
  network->num_h_layers = n_hidden_layers;
  network->neurons_per_h_layer = n_neurons_per_hlayer;

  // -- allocate memory for layers
  network->i_layer = (Neuron *)malloc(sizeof(Neuron) * n_i_neurons);
  network->o_layer = (Neuron *)malloc(sizeof(Neuron) * n_o_neurons);
  network->h_layers = (Neuron **)malloc(sizeof(Neuron *) * n_hidden_layers);
  for (int L = 0; L < n_hidden_layers; L++) {
    network->h_layers[L] =
        (Neuron *)malloc(sizeof(Neuron) * n_neurons_per_hlayer);
  }

  // -- init neurons (set bias input to 1, all else to 0), all
  // input layer: as many neurons as inputs to the network
  // input neuron: 1 input
  for (int i = 0; i < n_i_neurons; i++) {
    network->i_layer[i].af = activation_function_type;
    network->i_layer[i].num_inputs = 1;
    // input neurons have 1 input (+1 as bias)
    network->i_layer[i].input_vals = (double *)malloc(sizeof(double) * (1 + 1));
    network->i_layer[i].weights = (double *)malloc(sizeof(double) * (1 + 1));

    // init neurons weights, values to 0
    network->i_layer[i].input_vals[0] = 1.0; // fixed bias multiplicator
    network->i_layer[i].weights[0] = 0.0;    // bias
    // network input
    network->i_layer[i].input_vals[1] = 0.0; // input neuron input = 0
    network->i_layer[i].weights[1] = 0.0;    // input neuron weight = 0

    network->i_layer[i].output = 0.0; // input neuron output = 0
  }                                 // i, input layer neuron number

  // output layer: as many neurons as specified as network outputs
  // output neuron: # of inputs = #of previous layer's neurons
  for (int o = 0; o < n_o_neurons; o++) {
    network->o_layer[o].af = activation_function_type;
    network->o_layer[o].num_inputs = n_neurons_per_hlayer;
    // output neurons have as many inputs as hidden layer neurons (+1 as bias)
    network->o_layer[o].input_vals =
        (double *)malloc(sizeof(double) * (n_neurons_per_hlayer + 1));
    network->o_layer[o].weights =
        (double *)malloc(sizeof(double) * (n_neurons_per_hlayer + 1));
    // init neuron's bias to 0
    network->o_layer[o].input_vals[0] = 1.0; // fixed bias multiplicator 1
    network->o_layer[o].weights[0] = 0.0;    // bias = 0
    // init neuron's weights, values to 0
    for (int i = 1; i < (n_neurons_per_hlayer + 1); i++) {
      network->o_layer[o].input_vals[i] = 0.0; // output neuron input = 0
      network->o_layer[o].weights[i] = 0.0;    // output neuron weight = 0
    }
   network->o_layer[o].output = 0.0;// output neuron output = 0
  }                                 // o, output layer neuron number

  // hidden layers
  // hidden Layer 0 - special case, as connected to input neurons
  for (int l = 0; l < n_neurons_per_hlayer; l++) {
    network->h_layers[0][l].af = activation_function_type;
    network->h_layers[0][l].num_inputs = n_i_neurons;
    // neurons have as many inputs as hidden layer neurons (+1 as bias)
    network->h_layers[0][l].input_vals =
        (double *)malloc(sizeof(double) * (n_i_neurons + 1));
    network->h_layers[0][l].weights =
        (double *)malloc(sizeof(double) * (n_i_neurons + 1));
    // init neuron's bias to 0
    network->h_layers[0][l].input_vals[0] = 1.0; // fixed bias multiplicator 1
    network->h_layers[0][l].weights[0] = 0.0;    // bias = 0

    // init neuron's weights, values to 0
    for (int i = 1; i < (n_i_neurons + 1); i++) {
      network->h_layers[0][l].input_vals[i] = 0.0; // neuron input = 0
      network->h_layers[0][l].weights[i] = 0.0;    // neuron weight = 0
    }
    network->h_layers[0][l].output = 0.0; // neuron output = 0
  }

  // hidden layers 1-...
  for (int L = 1; L < n_hidden_layers; L++) {
    for (int l = 0; l < n_neurons_per_hlayer; l++) {
      network->h_layers[L][l].af = activation_function_type;
      network->h_layers[L][l].num_inputs = n_neurons_per_hlayer;
      // neurons have as many inputs as hidden layer neurons (+1 as bias)
      network->h_layers[L][l].input_vals =
          (double *)malloc(sizeof(double) * (n_neurons_per_hlayer + 1));
      network->h_layers[L][l].weights =
          (double *)malloc(sizeof(double) * (n_neurons_per_hlayer + 1));
      // init neuron's bias to 0
      network->h_layers[L][l].input_vals[0] = 1.0; // fixed bias multiplicator 1
      network->h_layers[L][l].weights[0] = 0.0;    // bias = 0

      // init neuron's weights, values to 0
      for (int i = 1; i < (n_neurons_per_hlayer + 1); i++) {
        network->h_layers[L][l].input_vals[i] = 0.0; // neuron input = 0
        network->h_layers[L][l].weights[i] = 0.0;    // neuron weight = 0
      }
      network->h_layers[L][l].output = 0.0; // neuron output = 0
    }                                       // l, hidden layer neuron number
  }                                         // L, layer number

  return network;
}

// Function to free memory allocated for the neural network
void NN_Network_free(NeuralNetwork *network) {
  for (int i = 0; i < network->num_inputs; i++) {
    free(network->i_layer[i].input_vals);
    free(network->i_layer[i].weights);
  }
  // hidden layers
  for (int L = 0; L < network->num_h_layers; L++) {
    for (int l = 0; l < network->neurons_per_h_layer; l++) {
      free(network->h_layers[L][l].input_vals);
      free(network->h_layers[L][l].weights);
    }
  }
  for (int L = 0; L < network->num_h_layers; L++) {
    free(network->h_layers[L]);
  }
  free(network->h_layers);
  // output layer
  for (int o = 0; o < network->num_outputs; o++) {
    Neuron n = network->o_layer[o];
    free(n.input_vals);
    free(n.weights);
  }
}

// Function to set input values for the input layer
void NN_Network_input_values_set(NeuralNetwork *network, double *inputValues) {
  for (int i = 0; i < network->num_inputs; i++) {
    // ...[0] = bias;
    network->i_layer[i].input_vals[1] = inputValues[i];
  }
}

// Function for forward propagation to calculate the output of the network
void NN_Network_propagate_forward(NeuralNetwork *network) {
  // [1] We assume input values have been set: in the input_vals of the
  // neurons in the i_layer neurons (ni->input_vals[1]).
  //
  // Next we "evaluate" the input layer: for each input neuron:
  // - we calculate the weighted sum (over all indizes incl. 0, the bias).
  // - we call the activation function on the weighted sum
  // - we store the result in the neuron's output.
  // - then we transfer the output to the inputs of the first hidden layer
  // (h_layers[0]):
  //   it is important to there store the input values from index 1 onwards,
  //   as input val[0] is always used for the bias in every neuron:
  //   - for each neuron we find in the hidden layer[0], we set our output
  //     as the hidden neuron's input [i+1].
  //     doing it right here helps to treat all neurons in all hidden layers
  //     the same in the next big step:
  //
  // [2]: Now we "evaluate" all the hidden layers: our neuron index is h.
  // - for all hidden layers L:
  //   - for all neurons in the hidden layer: we can do "[1]",
  //     with the only "difference": we store the neuron's output as
  //     input_vals[h+1] in all the next hidden layers neurons:
  //     h_layers[L+1][h].input_vals[h+1] .
  //     of course we can not do that in the last hidden layer.
  //     in the last hidden layer we store our output in all output layer's
  //     neuron's input_vals[h+1] instead.
  //
  // [3]: Now we can evaluate the output layer: just calculate the weighted
  // sum and call the activation function. this is the output of each output
  // neuron.

  // for all input neurons
  for (int i = 0; i < network->num_inputs; i++) {
    Neuron *neuron = &network->i_layer[i]; // tmp var for debugging
    NN_Neuron_process(neuron);
    // forward to hidden layer
    for (int h = 0; h < network->neurons_per_h_layer; h++) {
      network->h_layers[0][h].input_vals[i + 1] = neuron->output;
    }
  }

  // hidden layers
  for (int L = 0; L < network->num_h_layers; L++) {
    for (int h = 0; h < network->neurons_per_h_layer; h++) {
      Neuron *neuron = &network->h_layers[L][h];
      NN_Neuron_process(neuron);

      // store in next layer or output layer
      if (L < (network->num_h_layers - 1)) {
        // store in next layer
        for (int l = 0; l < network->neurons_per_h_layer; l++) {
          network->h_layers[L + 1][l].input_vals[h + 1] = neuron->output;
        }
      } else {
        // store in output layer
        for (int l = 0; l < network->num_outputs; l++) {
          network->o_layer[l].input_vals[h + 1] = neuron->output;
        }
      } // store in hidden or output layer
    }   // h
  }     // L

  // output layer: for all output neurons
  for (int n = 0; n < network->num_outputs; n++) {
    Neuron *neuron = &network->o_layer[n]; // create a tmp var for debging
    NN_Neuron_process(neuron);
  }
}

void NN_Neuron_dump(Neuron *neuron) {
  Neuron n = *neuron;
  // inputs
  printf("I[%d]:", n.num_inputs);
  for (int i = 0; i < n.num_inputs; i++) {
    printf("%.2f", n.input_vals[i + 1]);
    if (i < (n.num_inputs - 1))
      printf(",");
  }
  // weights
  printf(" W[%d]:", n.num_inputs);
  for (int i = 0; i < n.num_inputs; i++) {
    printf("%.2f", n.weights[i + 1]);
    if (i < (n.num_inputs - 1))
      printf(",");
  }
  // bias:
  printf(" B:%.2f", n.weights[0]);
  // output
  printf(" O:%.2f\n", n.output);
}

void NN_Network_dump(NeuralNetwork *network) {
  printf("NN %p: #I:%d, #O:%d, #L:%d Lsz:%d\n", network,
         network->num_inputs, network->num_outputs, network->num_h_layers,
         network->neurons_per_h_layer);
  // inputs
  printf("  LI: %d neurons\n", network->num_inputs);
  for (int i = 0; i < network->num_inputs; i++) {
    printf("    N%d: ", i);
    NN_Neuron_dump(&network->i_layer[i]);
  }
  // hidden layers
  for (int L = 0; L < network->num_h_layers; L++) {
    printf("  LH%d: %d neurons\n", L, network->neurons_per_h_layer);
    for (int l = 0; l < network->neurons_per_h_layer; l++) {
      printf("    N%d/%d: ", L, l);
      Neuron n = network->h_layers[L][l];
      NN_Neuron_dump(&n);
    }
  }
  // outputs
  printf("  LO: %d neurons\n", network->num_outputs);
  for (int o = 0; o < network->num_outputs; o++) {
    printf("    N%d: ", o);
    NN_Neuron_dump(&network->o_layer[o]);
  }
  printf("\n");
}
