// xor nn, 2024 M64. Schallner <mario.a.schallner@gmail.com>
// gcc nn_neuralnet.c nn_evolution_ga.c nn_timing.c implementation_xor.c -lm
// --debug -Ofast -Wall -o xor

#include "implementation_xor.h"
#include "nn_neuralnet.h"
#include "nn_timing.h"
#include <stdio.h>

void colorprintf(int intensity, const char *f, ...) {
  printf("\x1b[38;2;%d;%d;%dm", intensity, intensity, intensity);
  va_list l;
  va_start(l, f);
  vprintf(f, l);
  va_end(l);
}

void xor_visualizer(int size, NeuralNetwork *n) {
  double v = 0.0f;
  double inputs[2];

  printf("0");
  v = 255;
  for (int x = 0; x < (size - 2); x++)
    colorprintf(v, "--");
  printf("-->1 I[0]\n");

  for (int y = 0; y < size; y++) {
    if (y == (size - 2))
      colorprintf(255, "v");
    else if (y == (size - 1))
      colorprintf(255, "1");
    else
      colorprintf(255, "|");

    for (int x = 0; x < size; x++) {
      // v = ( ((double)x) ) / ( (double)size ) * 255.0;
      inputs[0] = ((double)x) / ((double)size) * 255.0;
      inputs[1] = ((double)y) / ((double)size) * 255.0;

      NN_Network_input_values_set(n, inputs);
      // n->o_layer[0].weights[0] = 0.5; // test
      NN_Network_propagate_forward(n);

      v = n->o_layer[0].output * 255;
      v = (v < 0) ? 0.0 : v;
      v = (v > 255.0) ? 255.0 : v;

      colorprintf(v, "**");
    }
    printf("\n");
  }
  colorprintf(255, "I[1]\n");
}

int main() {
  unsigned long ts1 = 0;
  unsigned long ts2 = 0;

  ts1 = get_timestamp();
  // 2 inputs, 1 output, 1 hidden layer, hidden layer size: 4, relU
  NeuralNetwork *network = NN_Network_initialize(2, 1, 1, 4, NN_AF_RELU);
  ts2 = get_duration_since(ts1);
  printf(" * initialization took: %lu usecs\n", ts2);

  ts1 = get_timestamp();
  NN_Network_propagate_forward(network);
  ts2 = get_duration_since(ts1);

  printf("\n");
  printf(" * forward propagation took: %lu usecs\n", ts2);

  printf("\n");
  printf(" * network dump:\n");
  NN_Network_dump(network);

  printf("\n");
  printf(" * input/output xy-visualizer\n");
  xor_visualizer(20, network);

  NN_Network_free(network);

  return 0;
}
