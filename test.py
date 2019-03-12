import tensorflow as tf
import keras
from keras.applications.vgg16 import VGG16
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, MaxPool2D, Conv2D, Flatten, Dropout, Activation
from keras.models import Model, model_from_json, load_model
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
from DataGenerator_for_knowledge_distillation import DG_for_kd
from Create_NN_graph import Create_NN_graph
import json
import pprint
import math

# from tensorflow.python import debug as tf_debug
# import keras.backend as K


def loss_for_knowledge_distillation(y_true, y_pred):
    return y_pred


# sess = K.get_session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# K.set_session(sess)

print('Wypłycanie sieci')

# inputs = Input(shape=(32, 32, 3))
# x = Flatten()(inputs)
# x = Dense(10)(x)
# x = Softmax()(x)
# shallowed_model = Model(inputs=inputs, outputs=x)
# shallowed_model.summary()

# wczytanie sieci
# original_model = CreateNN.create_VGG16_for_CIFAR10()
# original_model = load_model('Zapis modelu/19-03-03 19-24/weights-improvement-238-0.88.hdf5')
#
# # layers_accuracy = []
# accuracy = np.loadtxt('Skutecznosc warstw.txt')
# layers_accuracy = np.zeros((1, 3))
# for i in range(0, int(len(accuracy)/3)-1):
#     layers_accuracy = np.append(layers_accuracy, [[accuracy[3*i], accuracy[3*i+1], accuracy[3*i+2]]], axis=0)    # Przekonwertowanie listy
#
# layers_accuracy = np.delete(layers_accuracy, 0, 0)
#
# margins = 0.015
# last_effective_layer = 0
# layers_to_remove = []
# for i in range(2, len(layers_accuracy)-1):                                      # Znalezienie warstw do usunięcia
#     difference = layers_accuracy[i-1][2] - layers_accuracy[last_effective_layer][2]
#
#     if difference < margins:
#         layers_to_remove.append(i)
#     else:
#         last_effective_layer = i - 1
#
# print('The following convolutional layers and their dependencies(ReLU, batch normalization)will be removed:',
#       layers_to_remove, '\n')
#
shallowed_model = load_model('Zapis modelu/19-03-11 20-03/weights-improvement-62-23.19.hdf5', custom_objects={'loss_for_knowledge_distillation': loss_for_knowledge_distillation})

# shallowed_model = NNModifier.remove_chosen_conv_layers(original_model, layers_to_remove)
# shallowed_model = NNModifier.add_loss_layer_for_knowledge_distillation(shallowed_model, num_classes=10)
SGD = SGD(lr=0.1, momentum=0.9, nesterov=True)
shallowed_model.compile(optimizer=SGD, loss=loss_for_knowledge_distillation)
# original_model.compile(SGD, loss='categorical_crossentropy', metrics=['accuracy'])


# Ustawienie ścieżki zapisu i stworzenie folderu jeżeli nie istnieje
scierzka_zapisu = 'Zapis modelu/' + str(datetime.datetime.now().strftime("%y-%m-%d %H-%M") + '/')
scierzka_zapisu_dir = os.path.join(os.getcwd(), scierzka_zapisu)
if not os.path.exists(scierzka_zapisu_dir):  # stworzenie folderu jeżeli nie istnieje
    os.makedirs(scierzka_zapisu_dir)

# Ustawienie ścieżki logów i stworzenie folderu jeżeli nie istnieje
scierzka_logow = 'log/' + str(datetime.datetime.now().strftime("%y-%m-%d %H-%M") + '/')
scierzka_logow_dir = os.path.join(os.getcwd(), scierzka_logow)
if not os.path.exists(scierzka_logow_dir):  # stworzenie folderu jeżeli nie istnieje
    os.makedirs(scierzka_logow_dir)

# Callback
learning_rate_regulation = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=7, verbose=1, mode='auto', cooldown=5, min_lr=0.0005)
csv_logger = keras.callbacks.CSVLogger('training.log')                          # Tworzenie logów
tensorBoard = keras.callbacks.TensorBoard(log_dir=scierzka_logow, write_graph=False)               # Wizualizacja uczenia
modelCheckPoint = keras.callbacks.ModelCheckpoint(                              # Zapis sieci podczas uczenia
    filepath=scierzka_zapisu + "/weights-improvement-{epoch:02d}-{loss:.2f}.hdf5", monitor='loss',
    save_best_only=True, period=7, save_weights_only=False)
earlyStopping = keras.callbacks.EarlyStopping(monitor='loss', patience=75)  # zatrzymanie uczenia sieci jeżeli
                                                                                # dokładność się nie zwiększa


params = {'dim': (32, 32),
          'batch_size': 128,
          'n_classes': 10,
          'n_channels': 3,
          'shuffle': True,
          'inputs_number': 3}

# training_gen = DataGenerator(x_data_name='x_train', y_data_name='y_train', data_dir='data/CIFAR10.h5', **params)
# validation_gen = DataGenerator(x_data_name='x_validation', y_data_name='y_validation', data_dir='data/CIFAR10.h5', **params)

training_gen = DG_for_kd(x_data_name='x_train', data_dir='data/CIFAR10.h5',
                         dir_to_weights='Zapis modelu/19-03-03 19-24/weights-improvement-238-0.88.hdf5', **params)
validation_gen = DG_for_kd(x_data_name='x_validation', data_dir='data/CIFAR10.h5',
                           dir_to_weights='Zapis modelu/19-03-03 19-24/weights-improvement-238-0.88.hdf5', **params)

shallowed_model.fit_generator(generator=training_gen,
                              use_multiprocessing=False,
                              workers=10,
                              epochs=1000,
                              callbacks=[csv_logger, tensorBoard, modelCheckPoint, earlyStopping, learning_rate_regulation],
                              initial_epoch=41
                              )
# shallowed_model.save('Zapis modelu/shallowed_model.h5')

shallowed_model = NNModifier.remove_loos_layer(shallowed_model)
shallowed_model.compile(optimizer=SGD, loss='categorical_crossentropy', metrics=['accuracy'])
Create_NN_graph.create_NN_graph(shallowed_model, name='temp')
[x_train, x_validation, x_test], [y_train, y_validation, y_test] = NNLoader.load_CIFAR10()
scores = shallowed_model.evaluate(x=x_test,
                                  y=y_test,
                                  verbose=1,
                                  )
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

scores = shallowed_model.evaluate(x=x_train,
                                  y=y_train,
                                  verbose=1,
                                  )
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# b = np.empty(4500)
# b = shallowed_model.predict_generator(generator=training_gen,
#                                       workers=4,
#                                       verbose=1,
#                                       use_multiprocessing=True)

# shallowed_model.summary()
# file = NNLoader.load('temp.txt')
# print('kot płot')
#
# count_nan = 0
# for i, table in enumerate(b):
#     for line in table:
#         if math.isnan(line):
#             count_nan += 1
#
# print('w tabeli jest', count_nan, 'NaN')






# file = open('temp.txt')
# json_str = file.read()
# python_obj = json.loads(json_str)
# formatted_str = pprint.pformat(python_obj, indent=4)
a = 1