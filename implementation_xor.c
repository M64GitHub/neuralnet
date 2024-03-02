// xor nn, 2024 M64. Schallner <mario.a.schallner@gmail.com>
// gcc nn_neuralnet.c nn_evolution_ga.c nn_timing.c implementation_xor.c -lm
// --debug -Ofast -Wall -o xor

#include "implementation_xor.h"
#include "nn_neuralnet.h"
#include "nn_timing.h"
#include <stdio.h>
#include <stdlib.h>

void cursor_reset() {
  printf("\x1b[0m"); // reset all modes
}

void cursor_home() {
  printf("\x1b[H"); // home pos
}

void term_clear() {
  printf("\x1b[2J"); // erase entire screen
  printf("\x1b[H");  // home pos
}

void colorprintf(int intensity, const char *f, ...) {
  printf("\x1b[48;2;%d;%d;%dm", intensity, intensity, intensity);
  va_list l;
  va_start(l, f);
  vprintf(f, l);
  va_end(l);
}

void xor_visualizer(int size, NeuralNetwork *n) {
  double v = 0.0f;
  double inputs[2];

  // -- output based on inputs
  printf("0");
  v = 255;
  for (int x = 0; x < (size - 2); x++)
    printf("--");
  printf("-->1 I[0]\n");

  for (int y = 0; y < size; y++) {
    if (y == (size - 2))
      printf("v");
    else if (y == (size - 1))
      printf("1");
    else
      printf("|");

    for (int x = 0; x < size; x++) {
      // v = ( ((double)x) ) / ( (double)size ) * 255.0;
      // inputs[0] = ((double)x) / ((double)size) * 255.0;
      // inputs[1] = ((double)y) / ((double)size) * 255.0;

      inputs[0] = ((double)x) / ((double)size);
      inputs[1] = ((double)y) / ((double)size);

      NN_Network_input_values_set(n, inputs);
      // n->o_layer[0].weights[0] = 0.5; // test
      NN_Network_propagate_forward(n);

      v = (int)(n->o_layer[0].output * 255.0);
      v = (v < 0) ? 0.0 : v;
      v = (v > 255.0) ? 255.0 : v;

      colorprintf(v, "%1.0f ", (v / 255.0));
      // colorprintf(v, "%.2f ", n->o_layer[0].output);
      printf("\x1b[0m"); // reset all modes
    }
    printf("\n");
  }
  printf("I[1]\n");
}

int main() {
  unsigned long ts1 = 0;
  unsigned long ts2 = 0;
  unsigned long ts3 = 0;
  unsigned long ts4 = 0;
  double o1, o2, o3, o4;

  term_clear();

  ts3 = get_timestamp();
  srand(ts1);
  // 2 inputs, 1 output, 2 hidden layer, hidden layer size: 4, relU
  NeuralNetwork *network = NN_Network_initialize(2, 1, 1, 2, NN_AF_RELU);
  ts2 = get_duration_since(ts1);
  printf(" * initialization took: %lu usecs\n", ts2);

  int iteration = 0;
  while (1) {
    iteration++;
    cursor_home();
    srand(ts1);
    NN_Network_randomize_weights(network);
    ts1 = get_timestamp();
    NN_Network_propagate_forward(network);
    ts2 = get_duration_since(ts1);
    printf("\n");
    printf(" * forward propagation took: %lu usecs\n", ts2);

    printf("\n");
    printf(" * network dump:\n");
    NN_Network_input_values_set(network, (double[]){0.0, 0.0});
    NN_Network_propagate_forward(network);
    o1 = network->o_layer[0].output;
    NN_Network_dump(network);

    NN_Network_input_values_set(network, (double[]){0.0, 1.0});
    NN_Network_propagate_forward(network);
    o2 = network->o_layer[0].output;
    // NN_Network_dump(network);

    NN_Network_input_values_set(network, (double[]){1.0, 0.0});
    NN_Network_propagate_forward(network);
    o3 = network->o_layer[0].output;
    // NN_Network_dump(network);

    NN_Network_input_values_set(network, (double[]){1.0, 1.0});
    NN_Network_propagate_forward(network);
    o4 = network->o_layer[0].output;
    NN_Network_dump(network);

    printf(" * input/output xy-visualizer\n");
    if (!(iteration % 1000))
      xor_visualizer(20, network);
    printf("\n");

    double delta = o4 - o1;
    // if (delta < 0)
    //   delta = -1.0 * delta;
    ts4 = get_duration_since(ts3);
    printf("iteration: %d, delta: %02.3f (time: %lu)\n", iteration, delta, ts4);
    if ((o1 < 0.5) && (o2 <= 1.0) && (o3 <= 1.0) && (o4 < 0.5) && (o2 >= 0.5) &&
        (o3 >= 0.5)) {
      xor_visualizer(20, network);
      break;
    }
  }

  NN_Network_free(network);

  return 0;
}
