# neuralnet

Artifical feed forward neural network for test and learning purposes, multi-class classification, (for now:) evolution-theory and genetic algorithm based learning. Hacking playground and deep dive into the concepts of modern ML and Ai.

## Motivation
The goal is to research and understand the concepts of modern ai better, by writing them myself and creating a playground for experimentation. One focus shall then lay in inspection of the later trained model, and trying to find out in what ways a model can be best analyzed, if, at all. I am thinking of writing a kind of visualizer / "debugger" of the trained model's activity when reacting to different inputs. I am curious if multiple (almost "isolatable") "brain regions" will emerge, that trigger specific output labels, or the model's "knowledge" will be completely un-analyzable.  

This is currently replicating lost work of my youth in C ;), and continueing with it. I think the first artificial neural network / evolution theory based learning combination I developed in C++, but not sure anymore.

## Status
Pre functional.  
Current model format will only have to store a weigth matrix, number of inputs and ouputs and probably an indicator which activation function was used (if I choose to keep it same for all neurons, not sure yet) ... 

 - structs for neurons, networks ... OK
 - initializeNetwork(NeuralNetwork *) ... OK
 - freeNetwork(NeuralNetwork *) ... OK
 - setInputValues(double[]) ... OK
 - dump functions for Neuron, NeuralNetwork ... OK

 - forwardPropagation(NeuralNetwork *) ...
 - loss function ...

 - learning process based on evolutions, genetic algorithms ...

## Outlook

The journey just begins. A translation to C++, zig and cuda-C is in my mind. Implement more classical training methods. Tensorflow compatibility of the model data is one of the main goals, too.
