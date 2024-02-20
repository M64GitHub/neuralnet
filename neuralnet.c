// artifical neural network for test and learning purposes. 2024, M64 Schallner
// <mario.a.schallner@gmail.com>
// A forward propagation artifical neural network (ANN).
// Evolution-theory and genetic algorithm based learning.
// Various activation functions to choose from.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h> // for gettimeofday

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
double sigmoid(double x) { 
  return 1.0 / (1.0 + exp(-x)); 
}

unsigned long get_timestamp() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  unsigned long r = 1000000 * tv.tv_sec + tv.tv_usec;
  return r;
}

unsigned long get_duration_since(unsigned long t1) {
  unsigned long t2 = get_timestamp();
  return t2 - t1;
}

// -- nn functions

// Function to initialize a neural network
NeuralNetwork *initializeNetwork(int n_i_neurons, int n_o_neurons,
                                 int n_hidden_layers,
                                 int n_neurons_per_hlayer) {
  NeuralNetwork *network = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));

  network->num_inputs = n_i_neurons;
  network->num_outputs = n_o_neurons;
  network->num_h_layers = n_hidden_layers;
  network->neurons_per_h_layer = n_neurons_per_hlayer;

  // Allocate memory for neurons / layers
  network->i_layer = (Neuron *)malloc(sizeof(Neuron) * n_i_neurons);
  network->o_layer = (Neuron *)malloc(sizeof(Neuron) * n_o_neurons);

  network->h_layers = (Neuron **)malloc(sizeof(Neuron *) * n_hidden_layers);
  for (int L = 0; L < n_hidden_layers; L++) {
    network->h_layers[L] =
        (Neuron *)malloc(sizeof(Neuron) * n_neurons_per_hlayer);
  }

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

    network->o_layer[o].output = 0.0; // output neuron output = 0
  }                                   // o, output layer neuron number

  // hidden layers
  for (int L = 0; L < n_hidden_layers; L++) {
    for (int l = 0; l < n_neurons_per_hlayer; l++) {
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
void freeNetwork(NeuralNetwork *network) {
  for (int i = 0; i < network->num_inputs; i++) {
    free(network->i_layer[i].input_vals);
    free(network->i_layer[i].weights);
  }

  for (int o = 0; o < network->num_outputs; o++) {
    free(network->o_layer[o].input_vals);
    free(network->o_layer[o].weights);
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
}

// Function to set input values for the input layer
void setInputValues(NeuralNetwork *network, double *inputValues) {
  for (int i = 0; i < network->num_inputs; i++) {
    // ...[0] = bias;
    network->i_layer[i].input_vals[1] = inputValues[i];
  }
}

// function
double weightedSum(Neuron *n) {
  double o = 0.0;

  for (int i = 0; i < n->num_inputs; i++) {
    o += n->input_vals[i] * n->weights[i];
  }

  return o;
}

// Function for forward propagation to calculate the output of the network
void forwardPropagation(NeuralNetwork *network) {
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

  double ws = 0.0;
  double o = 0.0;
  // for all input neurons
  for (int i = 0; i < network->num_inputs; i++) {
    ws = weightedSum(&network->i_layer[i]);
    o = sigmoid(ws);
    network->i_layer[i].output = o;
    for (int h = 0; h < network->neurons_per_h_layer; h++) {
      network->h_layers[0][h].input_vals[i + 1] = o;
    }
  }

  // hidden layers
  for (int L = 0; L < network->num_h_layers; L++) {
    for (int h = 0; h < network->neurons_per_h_layer; h++) {
      ws = weightedSum(&network->h_layers[L][h]);
      o = sigmoid(ws);
      network->h_layers[L][h].output = o;

      // store in next layer or output layer
      if(L < (network->num_h_layers - 1)) {
        // store in next layer
        for (int l = 0; l < network->neurons_per_h_layer; l++) {
          network->h_layers[L + 1][l].input_vals[h + 1] = o;
        }
      } else {
        // store in output layer
        for (int l = 0; l < network->num_outputs; l++) {
          network->o_layer[l].input_vals[h + 1] = o;
        }
      } // store in hidden or output layer
    } // h
  }   // L

  // output layer: for all output neurons
  for (int n = 0; n < network->num_outputs; n++) {
    ws = weightedSum(&network->o_layer[n]);
    o = sigmoid(ws);
    network->o_layer[n].output = o;
  }
}

void dumpNeuron(Neuron *neuron) {
  Neuron n = *neuron;
  // inputs
  printf("i{");
  for (int i = 0; i < n.num_inputs; i++) {
    printf("%.2f", n.input_vals[i + 1]);
    if (i < (n.num_inputs - 1))
      printf(",");
  }
  printf("}/");

  // weights
  printf("w{");
  for (int i = 0; i < n.num_inputs; i++) {
    printf("%.2f", n.weights[i + 1]);
    if (i < (n.num_inputs - 1))
      printf(",");
  }
  printf("}/");

  // bias:
  printf("b:%.2f/", n.weights[0]);

  // output
  printf("o:%.2f}", n.output);
}

// Function to dump values of an nn
void dumpNetwork(NeuralNetwork *network) {
  printf("dumping network %p: #i:%d, #o:%d, #L:%d Lsz:%d\n", network,
         network->num_inputs, network->num_outputs, network->num_h_layers,
         network->neurons_per_h_layer);

  // inputs
  printf("LI:{");
  for (int i = 0; i < network->num_inputs; i++) {
    // printf("%f", network->i_layer[i].input_vals[1]); // 0 is bias
    printf("N%d:", i);
    dumpNeuron(&network->i_layer[i]);
    if (i < (network->num_inputs - 1))
      printf(", ");
  }
  printf("}\n");

  // hidden layers
  for (int L = 0; L < network->num_h_layers; L++) {
    printf("LH[%d]:{", L);
    for (int l = 0; l < network->neurons_per_h_layer; l++) {
      printf("N%d:", l);
      Neuron n = network->h_layers[L][l];
      dumpNeuron(&n);
      if (l < (network->neurons_per_h_layer - 1))
        printf(", ");
    }
    printf("}\n");
  }

  // outputs
  printf("LO:{");
  for (int o = 0; o < network->num_outputs; o++) {
    // printf("%.2f", network->o_layer[o].output); // 0 is bias
    printf("N%d:", o);
    dumpNeuron(&network->o_layer[o]);
    if (o < (network->num_outputs - 1))
      printf(", ");
  }
  printf("}\n");
}

int main() {
  unsigned long ts1 = 0;
  unsigned long ts2 = 0;

  ts1 = get_timestamp();
  // 2 inputs, 1 output, 1 hidden layer, layer size: 3
  NeuralNetwork *network = initializeNetwork(2, 1, 2, 3);
  printf(" * initialization took: %lu usecs\n", get_duration_since(ts1));

  double i_vals[] = {0.1, 0.2};

  ts1 = get_timestamp();
  setInputValues(network, i_vals);
  printf(" * set input vals took: %lu usecs\n", get_duration_since(ts1));
  dumpNetwork(network);

  ts1 = get_timestamp();
  forwardPropagation(network);
  printf(" * forward propagation took: %lu usecs\n", get_duration_since(ts1));
  dumpNetwork(network);

  freeNetwork(network);

  return 0;
}
