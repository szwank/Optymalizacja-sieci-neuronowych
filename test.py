import tensorflow as tf
import keras
from keras.applications.vgg16 import VGG16
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, MaxPool2D, Conv2D, Flatten, Dropout, Activation
from keras.models import Model, model_from_json
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras.layers import Softmax
import numpy as np
from keras.utils import plot_model
import datetime
import os
from NNModifier import NNModifier
from NNLoader import NNLoader
from CreateNN import CreateNN
import json
import pprint


print('Wypłycanie sieci')
# wczytanie sieci
original_model = CreateNN.create_VGG16_for_CIFAR10()
layers_accuracy = []

accuracy = np.loadtxt('Skutecznosc warstw.txt')
layers_accuracy = np.zeros((1, 3))
for i in range(0, int(len(accuracy)/3)-1):
    layers_accuracy = np.append(layers_accuracy, [[accuracy[3*i], accuracy[3*i+1], accuracy[3*i+2]]], axis=0)    # Przekonwertowanie listy

layers_accuracy = np.delete(layers_accuracy, 0, 0)

margins = 0.015
last_effective_layer = 0
layers_to_remove = []
for i in range(2, len(layers_accuracy)-1):                                      # Znalezienie warstw do usunięcia
    difference = layers_accuracy[i-1][2] - layers_accuracy[last_effective_layer][2]

    if difference < margins:
        layers_to_remove.append(i)
    else:
        last_effective_layer = i - 1

print('The following convolutional layers and their dependencies(ReLU, batch normalization)will be removed:',
      layers_to_remove, '\n')

shallowed_model = NNModifier.remove_chosen_conv_layers(original_model, layers_to_remove)
# shallowed_model.summary()
# file = NNLoader.load('temp.txt')


# file = open('temp.txt')
# json_str = file.read()
# python_obj = json.loads(json_str)
# formatted_str = pprint.pformat(python_obj, indent=4)
a = 1