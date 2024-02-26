#include "neuralnet.h"
#include "timing.h"
#include <stdio.h>

int main() {
  unsigned long ts1 = 0;
  unsigned long ts2 = 0;

  ts1 = get_timestamp();
  // 2 inputs, 1 output, 1 hidden layer, layer size: 3, activation function:
  // sigmoid
  // NeuralNetwork *network = initializeNetwork(2, 1, 1, 3, NN_AF_SIGMOID);
  // NeuralNetwork *network = initializeNetwork(10, 3, 2, 5, NN_AF_SIGMOID);
  ts2 = get_duration_since(ts1);

  // potential values for MNIST
  NeuralNetwork *network = initializeNetwork(784, 10, 1, 128, NN_AF_RELU);
  printf(" * initialization took: %lu usecs\n", ts2);

  // double i_vals[] = {0.1, 0.2};
  // ts1 = get_timestamp();
  // setInputValues(network, i_vals);
  // printf(" * set input vals took: %lu usecs\n", get_duration_since(ts1));
  // dumpNetwork(network);

  ts1 = get_timestamp();
  forwardPropagation(network);
  ts2 = get_duration_since(ts1);
  printf(" * forward propagation took: %lu usecs\n", ts2);
  // dumpNetwork(network);

  freeNetwork(network);

  return 0;
}
