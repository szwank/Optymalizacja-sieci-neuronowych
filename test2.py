import os
from keras.preprocessing.image import ImageDataGenerator
from GeneratorStorage.GeneratorDataLoaderFromMemory import GeneratorDataLoaderFromMemory
from GeneratorStorage.GeneratorsFlowStorage import GeneratorsFlowStorage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from NNLoader import NNLoader
import numpy as np
import matplotlib
import json
import os
from keras.layers import Dense, BatchNormalization, LeakyReLU, Activation, Input, Flatten, Softmax
from keras.models import Model, load_model
from Create_NN_graph import Create_NN_graph
from NNLoader import NNLoader
import keras.backend as K
import math

def softmax(arg):
    anser = []
    sum_ = 0
    for i in range(len(arg)):
        sum_ += math.exp(arg[i])

    for i in range(len(arg)):
        a = math.exp(arg[i])
        anser.append(a/sum_)
    return anser

training_data = NNLoader.load_CIFAR10()
model = load_model('Zapis modelu/VGG16-CIFAR10-0.94acc.hdf5', compile=False)
model.layers.pop()
logits = model.layers[-1].output
model = Model(model.inputs, logits)

data = []
for temperature in range(1, 15):
    prediction = model.predict_on_batch(training_data.get_training_inputs()[1:2])
    prediction = np.ndarray.tolist(prediction)[0]

    for i in range(len(prediction)):
        prediction[i] = prediction[i] / temperature

    prediction = softmax(prediction)
    print(prediction)
    data.append(max(prediction))

plt.figure(1)
plt.plot(np.arange(1, 15), data, 'bo')
plt.ylabel('Prawdopodobieństwo najbardziej prawdopodobnej odpowiedzi')
plt.xlabel('T- wartość temperatury')
plt.show()