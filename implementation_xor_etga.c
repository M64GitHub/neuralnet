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
  NN_Network_input_values_set(network, (double[]){0.0, 0.0});
  NN_Network_propagate_forward(network);
  *o1 = (int)(network->o_layer[0].output + 0.5f);
  printf(" * network dump I[]: [0,0] -> O1: %d\n", *o1);
  NN_Network_dump(network);

  NN_Network_input_values_set(network, (double[]){0.0, 1.0});
  NN_Network_propagate_forward(network);
  *o2 = (int)(network->o_layer[0].output + 0.5f);
  printf(" * network dump I[]: [0,1]-> O2: %d\n", *o2);
  NN_Network_dump(network);

  NN_Network_input_values_set(network, (double[]){1.0, 0.0});
  NN_Network_propagate_forward(network);
  *o3 = (int)(network->o_layer[0].output + 0.5f);
  printf(" * network dump I[]: [1,0] -> O3: %d\n", *o3);
  NN_Network_dump(network);

  NN_Network_input_values_set(network, (double[]){1.0, 1.0});
  NN_Network_propagate_forward(network);
  *o4 = (int)(network->o_layer[0].output + 0.5f);
  printf(" * network dump I[]: [1,1] -> O4: %d\n", *o4);
  NN_Network_dump(network);
}

double xor_fitness(NeuralNetwork *network) {
  double F = 0.0f;
  double fitness = 0.0f;
  double output = 0.0f;
  double expect = 0.0f;
  double deviation = 0.0f;
  double deviation_scale = 1.0;
  int wrongs = 0;

  // 0, 0
  expect = 0.0f;
  fitness = 0.0f;
  NN_Network_input_values_set(network, (double[]){0.0, 0.0});
  NN_Network_propagate_forward(network);
  output = (network->o_layer[0].output);
  deviation = output - expect;
  if (deviation < 0.0)
    deviation = -deviation;
  fitness = deviation * deviation_scale;
  F -= fitness;
  if (deviation >= 0.5)
    wrongs++;
  printf("fitness 0,0: o:%.2f,f:%.4f |  d:%.4f, d*s:%.4f, w:%d\n", output,
         fitness, deviation, deviation * deviation_scale, wrongs);

  // 0, 1
  expect = 1.0f;
  fitness = 0.0f;
  NN_Network_input_values_set(network, (double[]){0.0, 1.0});
  NN_Network_propagate_forward(network);
  output = (network->o_layer[0].output);
  deviation = output - expect;
  if (deviation < 0.0)
    deviation = -deviation;
  fitness = deviation * deviation_scale;
  F -= fitness;
  if (deviation >= 0.5)
    wrongs++;
  printf("fitness 1,0: o:%.2f, f:%.4f|  d:%.4f, d*s:%.4f, w:%d\n", output,
         fitness, deviation, deviation * deviation_scale, wrongs);

  // 1, 0
  expect = 1.0f;
  fitness = 0.0f;
  NN_Network_input_values_set(network, (double[]){1.0, 0.0});
  NN_Network_propagate_forward(network);
  output = (network->o_layer[0].output);
  deviation = output - expect;
  if (deviation < 0.0)
    deviation = -deviation;
  fitness = deviation * deviation_scale;
  F -= fitness;
  if (deviation >= 0.5)
    wrongs++;
  printf("fitness 0,1: o:%.2f, f:%.4f |  d:%.4f, d*s:%.4f, w:%d\n", output,
         fitness, deviation, deviation * deviation_scale, wrongs);

  // 1, 1
  expect = 0.0f;
  fitness = 0.0f;
  NN_Network_input_values_set(network, (double[]){1.0, 1.0});
  NN_Network_propagate_forward(network);
  output = (network->o_layer[0].output);
  deviation = output - expect;
  if (deviation < 0.0)
    deviation = -deviation;
  fitness = deviation * deviation_scale;
  F -= fitness;
  if (deviation >= 0.5)
    wrongs++;
  printf("fitness 1,1: o:%.2f, f:%.4f |  d:%.4f, d*s:%.4f, w:%d\n", output,
         fitness, deviation, deviation * deviation_scale, wrongs);

  F -= wrongs;
  printf("fitness final: %.4f\n", F);

  return F;
}

int main() {
  unsigned long ts1 = 0;
  unsigned long ts2 = 0;
  unsigned long ts3 = 0; // overall time
  unsigned long ts4 = 0; //
  int o1, o2, o3, o4;
  char buf4k[4096];

  // term_clear();
  printf("===============================================================\n");
  printf("XOR Evolution Theory / Genetic Algorithm based learning\n");
  printf("===============================================================\n");

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

  P = NN_Population_initialize(10, network);
  NN_Population_list_individuals(P);
  NN_Population_run_forward_propagation(P);
  NN_Population_dump_individuals(P);

  printf(" * press ENTER to continue ...");
  fread(buf4k, 1, 1, stdin);

  int iteration = 0;
  while (1) {
    iteration++;
    // cursor_home();
    term_clear();
    ts1 = get_timestamp();
    srand(ts1);
    NN_Network_randomize_weights(network);
    NN_Population_run_forward_propagation(P);
    dump_network4x(&o1, &o2, &o3, &o4, network);
    ts4 = get_duration_since(ts3);
    printf("iteration: %d, (time: %lu)\n", iteration, ts4);
    // printf("O1: %d, O2: %d, O3: %d, O4: %d, fitness: %f\n", o1, o2, o3, o4,
    //        xor_fitness(network));
    if (!(iteration % 1000))
      xor_visualizer(20, network);
    if ((o1 == 0) && (o2 == 1) && (o3 == 1) && (o4 == 0)) {
      term_clear();
      printf(" * forward propagation took: %lu usecs\n", ts2);
      printf("iteration: %d, (time: %lu)\n", iteration, ts4);
      dump_network4x(&o1, &o2, &o3, &o4, network);
      xor_visualizer(20, network);
      // printf("O1: %d, O2: %d, O3: %d, O4: %d, fitness: %f\n", o1, o2, o3, o4,
      //        xor_fitness(network));
      break;
    }

    // printf(" * press ENTER to continue ...");
    // fread(buf4k, 1, 1, stdin);
  }

  NN_Network_free(network);

  return 0;
}
