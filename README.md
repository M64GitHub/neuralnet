# neuralnet

Artificial Feedforward Evolutionary/Genetic Neural Network: An Adaptable Model for Test and Learning Purposes in Multi-class Classification or Nonlinear Regression. Evolution-theory and genetic algorithm based learning. Hacking playground and deep dive into the concepts of modern ML and Ai.

## Motivation
The goal is to research and better understand the concepts of modern ML, by writing them myself and creating a playground for experimentation. One focus shall then lay in inspection of the later trained model, and trying to find out in what ways a model can be best analyzed, if, at all. I am thinking of writing a kind of visualizer / "debugger" of the trained model's activity when reacting to different inputs. I am curious if multiple (almost "isolatable") "brain regions" will emerge, that trigger specific output labels, or the model's "knowledge" will be completely un-analyzable.  

This is currently replicating lost work of my youth in C ;), and continueing with it. I think the first artificial neural network / evolution theory based learning combination I developed in C++, but not sure anymore.

## Status
- Overall: Pre functional
- Neural net computations: functional
- Training: /
- Evolutionary / genetic algorithms: /

## Specification
Current neuron format:
 - number of inputs
 - inputs[]
 - weights[]
 - output

Current model format:
 - number of inputs (input neurons, each w/ 1 input)
 - number of hidden layers
 - number of neurons per hidden layer
 - number of outputs
 - input layer
 - hidden layers[]
 - output layer

The structure of the final model is dynamically created, derived from the net's "number of ..." parameters.

### Implementation Notes
 - structs for neurons, networks ... OK
 - initializeNetwork(NeuralNetwork *) ... OK
 - freeNetwork(NeuralNetwork *) ... OK
 - setInputValues(double[]) ... OK
 - dump functions for Neuron, NeuralNetwork ... OK
 - forwardPropagation(NeuralNetwork *) ... OK
 - learning process based on evolution- and genetic algorithms ...
 - progress visualisation ...
 - export to any industry standard TensorFlow/Torch ...
   
#### Model specific
 - data import
 - loss/fitness function ...

## Outlook

The journey just begins. A translation to C++, zig and cuda-C is in my mind, for speed comparisons. Implement more classical training methods (esp for nonlinear regression). Exporting to Tensorflow- or PyTorch-compatible model formats is one of the main goals, too.
