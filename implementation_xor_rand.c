// xor nn, 2024 M64. Schallner <mario.a.schallner@gmail.com>
// gcc nn_neuralnet.c nn_evolution_ga.c nn_timing.c implementation_xor.c -lm
// --debug -Ofast -Wall -o xor

#include "implementation_xor.h"
#include "nn_evolution_ga.h"
#include "nn_neuralnet.h"
#include "nn_timing.h"
#include <stdio.h>
#include <stdlib.h>

Population *P;

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
  double inputs[2];
  int out = 0;
  int v = 0;

  // -- output based on inputs
  printf("0");
  v = 255;
  for (int x = 0; x <= (size - 2); x++)
    printf("--");
  printf("-->1 I[0]\n");

  for (int y = 0; y <= size; y++) {
    if (y == (size - 1))
      printf("v");
    else if (y == (size))
      printf("1");
    else
      printf("|");

    for (int x = 0; x <= size; x++) {
      inputs[0] = ((double)x) / ((double)size);
      inputs[1] = ((double)y) / ((double)size);

      NN_Network_input_values_set(n, inputs);
      NN_Network_propagate_forward(n);
      out = (int)((n->o_layer[0].output) + 0.5f);
      v = (int)((n->o_layer[0].output * 255.0) + 0.5f);
      colorprintf(v, " %1d", out);
      printf("\x1b[0m"); // reset all modes
    }
    printf("\n");
  }
  printf("I[1]\n");
}

void dump_network4x(int *o1, int *o2, int *o3, int *o4,
                    NeuralNetwork *network) {
  printf(" * network dump I[]: [0,0] -> O1:\n");
  NN_Network_input_values_set(network, (double[]){0.0, 0.0});
  NN_Network_propagate_forward(network);
  *o1 = (int)(network->o_layer[0].output + 0.5f);
  NN_Network_dump(network);

  printf(" * network dump I[]: [0,1]-> O2:\n");
  NN_Network_input_values_set(network, (double[]){0.0, 1.0});
  NN_Network_propagate_forward(network);
  *o2 = (int)(network->o_layer[0].output + 0.5f);
  NN_Network_dump(network);

  printf(" * network dump I[]: [1,0] -> O3:\n");
  NN_Network_input_values_set(network, (double[]){1.0, 0.0});
  NN_Network_propagate_forward(network);
  *o3 = (int)(network->o_layer[0].output + 0.5f);
  NN_Network_dump(network);

  printf(" * network dump I[]: [1,1] -> O4:\n");
  NN_Network_input_values_set(network, (double[]){1.0, 1.0});
  NN_Network_propagate_forward(network);
  *o4 = (int)(network->o_layer[0].output + 0.5f);
  NN_Network_dump(network);
}

int main() {
  unsigned long ts1 = 0;
  unsigned long ts2 = 0;
  unsigned long ts3 = 0; // overall time
  unsigned long ts4 = 0; //
  int o1, o2, o3, o4;
  char buf4k[4096];

  term_clear();

  ts3 = get_timestamp();
  srand(ts3);
  // 2 inputs, 1 output, 2 hidden layer, hidden layer size: 4, relU
  ts1 = get_timestamp();
  NeuralNetwork *network = NN_Network_initialize(2, 1, 1, 2, NN_AF_RELU);
  ts2 = get_duration_since(ts1);
  printf(" * initialization took: %lu usecs\n", ts2);
  ts1 = get_timestamp();
  NN_Network_propagate_forward(network);
  ts2 = get_duration_since(ts1);
  printf("\n");
  printf(" * forward propagation took: %lu usecs\n", ts2);

  int iteration = 0;
  while (1) {
    iteration++;
    cursor_home();
    ts1 = get_timestamp();
    srand(ts1);
    NN_Network_randomize_weights(network);
    NN_Population_run_forward_propagation(P);
    dump_network4x(&o1, &o2, &o3, &o4, network);
    ts4 = get_duration_since(ts3);
    printf("iteration: %d, (time: %lu)\n", iteration, ts4);
    if (!(iteration % 1000))
      xor_visualizer(20, network);
    if ((o1 == 0) && (o2 == 1) && (o3 == 1) && (o4 == 0)) {
      term_clear();
      printf(" * forward propagation took: %lu usecs\n", ts2);
      printf("iteration: %d, (time: %lu)\n", iteration, ts4);
      dump_network4x(&o1, &o2, &o3, &o4, network);
      xor_visualizer(20, network);
      printf("O1: %d, O2: %d, O3: %d, O4: %d\n", o1, o2, o3, o4);
      break;
    }
  }

  NN_Network_free(network);

  return 0;
}
