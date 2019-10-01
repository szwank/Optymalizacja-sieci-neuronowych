## Description

The program allows to train and optimize the structure of convolutional neural networks by removing redundant convolutional layers and filters in a convolutional layer. The program analyzes the influence of the accuracy of each convolutional layer and each filter in convolutional layers. Based on the analysis, individual filters and whole layers could be removed from the neural network structure by the assumed heuristic. Later to recover accuracy of the optimized neural network, a process called knowledge distillation is used. It this process optimized neural network is trying to approximate all answers, even from wrong classes, of bigger neural network.
Optimization barely lowers or even sometimes increases the accuracy of the optimized neural network. The number of excise parameters depends on settings of optimization, network architecture, and data set that was used to train the network. In some cases, even ¾ of network parameters can be excised with no penalty on accuracy. Network optimized by my algorithm achieved almost state of art result on classifying CIFAR-10 dataset with score 98.8% on test dataset.
I created this application for my Master’s thesis.

## Technologies used

- Python
- Keras (TensorFlow backend)

## Prerequisites

- CUDA toolkit 9.2
- cuDNN toolkit
- Tensorflow-gpu 1.12.0
- Keras 2.2.4

- dataset photos of skin changes
