#include "nn_neuralnet.h"
#include "nn_timing.h"
#include <stdio.h>

int main() {
  unsigned long ts1 = 0;
  unsigned long ts2 = 0;

  ts1 = get_timestamp();
  // 2 inputs, 1 output, 4 hidden layers, layer size: 10, relU
  NeuralNetwork *network = NN_Network_initialize(2, 1, 4, 10, NN_AF_RELU);
  // NN_Network_input_values_set(network, (double[]){2.0, 3.0});

  // ts1 = get_timestamp();
  // potential values for MNIST
  // NeuralNetwork *network = NN_Network_initialize(784, 10, 1, 128, NN_AF_RELU);
  ts2 = get_duration_since(ts1);
  printf(" * initialization took: %lu usecs\n", ts2);

  ts1 = get_timestamp();
  NN_Network_propagate_forward(network);
  ts2 = get_duration_since(ts1);
  printf(" * forward propagation took: %lu usecs\n", ts2);
  // NN_Network_dump(network);

  NN_Network_free(network);

  return 0;
}
