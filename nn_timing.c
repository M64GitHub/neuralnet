// artifical neural network for test and learning purposes. 2024, M64 Schallner
// <mario.a.schallner@gmail.com>
#include "nn_timing.h"

unsigned long get_timestamp() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  unsigned long r = 1000000 * tv.tv_sec + tv.tv_usec;
  return r;
}

unsigned long get_duration_since(unsigned long t1) {
  unsigned long t2 = get_timestamp();
  return t2 - t1;
}

