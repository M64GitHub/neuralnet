// artifical neural network for test and learning purposes. 2024, M64 Schallner
// <mario.a.schallner@gmail.com>
#ifndef __TIMING_H__
#define __TIMING_H__
#include <sys/time.h> // for gettimeofday

unsigned long get_timestamp();
unsigned long get_duration_since(unsigned long t1);

#endif
