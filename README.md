# neuralnet

Artificial feedforward evolutionary/genetic neural network: An adaptable generic model for test and learning purposes. Evolution-theory and genetic algorithm based learning vs classic backpropagation later on. Hacking playground and deep dive into the concepts of modern ML.


## Motivation
The objective is to explore and deepen understanding of contemporary machine learning concepts by implementing them from scratch and establishing a platform for experimentation. One key area of focus will involve analyzing the trained models to determine the most effective methods for assessment, if any. I plan to develop a visualizer to observe the trained model's behavior in response to various inputs. 

#### Why C?
Currently, this project involves recreating work I undertook in my youth using C++, and expanding upon it. I opted for C at this time because it facilitates future translation, such as converting the code into CUDA kernels and utilizing the existing structs as they are. Additionally, the C code can be readily translated into C++ classes. Furthermore, translating it to Zig is likely easier than starting from a C++ code base.


## Status
- Overall: Pre functional
- Neural net computations: functional
- Training: in progress
- Evolutionary / genetic algorithms: in progress
- Visualizing: in progress
- Debugging: in progress
- Model Data import / export: /
- Add optional convolutional layer type, filters, pooling: /

First real test results show XOR can already be found without training, just by randomizing the network's weigts only ;) - examples:


![image](https://github.com/M64GitHub/neuralnet/assets/84202356/d1d7ff69-fc3f-46b7-9e1d-ff6d5315a2aa)

![image](https://github.com/M64GitHub/neuralnet/assets/84202356/a76551a1-3253-47b4-ba82-11726e3b4bdd)


## Specification
I opted for a format, where I bind the connection weights to their "end point", into the structure of the connected neuron on the receivers end. It is the neuron that processes the signal through it's activation function, so it made sense for me to store it there. As a consequence, I do not have a specific "layer structure", layers are defined as linear flat arrays of neurons.  
Also I wanted to be able to have all elements for calculations in one place. Mainly for ease of inspection / visualisation. So I chose to store a neuron's connection input also directly in the receivers neuron structure, additional to the input weights. Hence I have all parameters for calculating a neuron's output in one place, the neuron itself I am currently looking at.

Current neuron format:
 - number of inputs
 - inputs[]
 - weights[]
 - output
 - type of activation function

weights[0] represents the bias.

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

The journey just begins. A translation to zig, ROCm and CUDA-C is on my mind, for speed comparisons. Implement more classical training methods (esp for nonlinear regression). Exporting to Tensorflow- or PyTorch-compatible model formats is one of the main goals, too.

### Build
For now:
```
gcc nn_neuralnet.c nn_evolution_ga.c nn_timing.c main.c -lm --debug -Ofast -Wall -o neuralnet
```

