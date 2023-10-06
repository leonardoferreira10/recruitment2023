# Recruitment Task 2023

To solve this task, you must create a fork of this repository on your GitHub account and work from there. When done, submit a pull request to the main branch so we can review your work. Feel free to create a draft pull request the moment you start solving the task, if you want.

## Module 1 - Implementing and training a neural network

### Requirements

- Python
    - PyTorch

### Tasks

In this module, the goal is to develop and train a neural network to identify hand-written digits (MNIST dataset) using PyTorch.

1. The provided architecture is a MLP with 3 fully-connected layers:
    - 784 neurons
    - 800 neurons
    - 10 neurons
    1. Why does the input layer have 784 neurons?
    2. Why does the output layer have 10 neurons?

2. The initial architecture is a good old Multi-Layer Perceptron (MLP). 
    1. Modify the architecture to be a Convolutional Neural Network (CNN) with the following layers:
        - Convolutional layer with 6 output channels, 5x5 kernel size, stride 1 and tanh activation
        - Average pooling layer with 6 output channels, 2x2 kernel size, stride 2 and tanh activation
        - Convolutional layer with 16 output channels, 5x5 kernel size, stride 1 and tanh activation
        - Average pooling layer with 16 output channels, 2x2 kernel size, stride 2 and tanh activation
        - Convolutional layer with 120 output channels, 5x5 kernel size, stride 1 and tanh activation
        - Fully-connected layer with 84 neurons and tanh activation
        - Fully-connected layer with 10 neurons and softmax activation
    2. The tanh and softmax activation functions are differentiable. Can you explain why not using non-differentiable functions?
    3. What changed in the results when comparing with the MLP?