# neuralnet

Artificial Feedforward Evolutionary/Genetic Neural Network: An Adaptable Model for Test and Learning Purposes in Multi-class Classification or Nonlinear Regression. Evolution-theory and genetic algorithm based learning. Hacking playground and deep dive into the concepts of modern ML and Ai.

## Motivation
The objective is to explore and deepen understanding of contemporary machine learning concepts by implementing them from scratch and establishing a platform for experimentation. One key area of focus will involve analyzing the trained models to determine the most effective methods for assessment, if any. I plan to develop a visualizer or "debugger" to observe the trained model's behavior in response to various inputs. I'm curious to see if distinct "brain regions" will emerge, each triggering specific output labels, or if the model's comprehension will remain opaque.

#### Why C?
Currently, this project involves recreating work I undertook in my youth using C++, and expanding upon it. I opted for C at this time because it facilitates future translation, such as converting the code into CUDA kernels and utilizing the existing structs as they are. Additionally, the C code can be readily translated into C++ classes. Furthermore, translating it to Zig is likely easier than starting from a C++ code base.


## Status
- Overall: Pre functional
- Neural net computations: functional
- Training: /
- Evolutionary / genetic algorithms: in progress
- Visualizing / debugging: /
- Model Data import / export: /
- Add optional convolutional layer type, filters, pooling: /

## Specification
Current neuron format:
 - number of inputs
 - inputs[]
 - weights[]
 - output
 - type of activation function

Current model format:
 - number of inputs (input neurons, each w/ 1 input)
 - number of hidden layers
 - number of neurons per hidden layer
 - number of outputs
 - input layer
 - hidden layers[]
 - output layer
 - type of activation functions for all neurons

The structure of the final model is dynamically created, derived from the net's "number of ..." parameters.

### Implementation Notes
 - structs for neurons, networks ... OK
 - initializeNetwork(NeuralNetwork *) ... OK
 - freeNetwork(NeuralNetwork *) ... OK
 - setInputValues(double[]) ... OK
 - dump functions for Neuron, NeuralNetwork ... OK
 - forwardPropagation(NeuralNetwork *) ... OK
 - learning process based on evolution- and genetic algorithms ... in progress
 - progress visualisation ...
 - export to any industry standard TensorFlow/Torch ...
   
#### Model specific
 - data import
 - loss/fitness function ...

## Outlook

The journey just begins. A translation to C++, zig and cuda-C is in my mind, for speed comparisons. Implement more classical training methods (esp for nonlinear regression). Exporting to Tensorflow- or PyTorch-compatible model formats is one of the main goals, too.

### Build
For now:
```
gcc neuralnet.c timing.c main.c -lm --debug -o neuralnet
```

