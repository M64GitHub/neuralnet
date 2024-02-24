#include <stdio.h>
#include "neuralnet.h"
#include "timing.h"

int main() {
  unsigned long ts1 = 0;
  unsigned long ts2 = 0;

  ts1 = get_timestamp();
  // 2 inputs, 1 output, 1 hidden layer, layer size: 3
  NeuralNetwork *network = initializeNetwork(2, 1, 2, 3, NN_AF_SIGMOID);
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
