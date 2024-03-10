# 2024 M64 Schallner, <mario.a.schallner@gmail.com>


all:
	gcc nn_neuralnet.c nn_evolution_ga.c nn_timing.c main.c -lm --debug -Ofast -Wall -o neuralnet
	gcc nn_neuralnet.c nn_evolution_ga.c nn_timing.c implementation_xor_rand.c -lm --debug -Ofast -Wall -o xor_rand
	gcc nn_neuralnet.c nn_evolution_ga.c nn_timing.c implementation_xor_etga.c -lm --debug -Ofast -Wall -o xor_etga

