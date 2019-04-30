import unittest
from keras.models import load_model, Model
from keras.optimizers import SGD
from keras.layers import Lambda
from CreateNN import CreateNN
import keras.backend as K
import h5py
import numpy as np
from DataGenerator_for_knowledge_distillation import DataGenerator_for_knowledge_distillation
import sys


class TestDataGeneratorForKnowledgeDistillation(unittest.TestCase):
    def test_if_data_is_generated_correctly(self):
        path_to_weights = 'Zapis modelu/VGG16-CIFAR10-0.94acc.hdf5'
        model = load_model(path_to_weights)
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

        params = {'dim': (32, 32, 3),
                  'batch_size': 128,
                  'number_of_classes': 10,
                  'shuffle': True,
                  'inputs_number': 3}

        training_gen = DataGenerator_for_knowledge_distillation(name_of_data_set_in_file='x_train',
                                                                path_to_h5py_data_to_be_processed='data/CIFAR10.h5',
                                                                path_to_weights=path_to_weights, **params)
        print('Testing logits:')
        h5f_ansers = h5py.File('temp/Generator_data.h5', 'r')
        h5f_input = h5py.File('data/CIFAR10.h5', 'r')
        optimizer = SGD(lr=0.1, momentum=0.9, nesterov=True)
        model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        for i in range(int(len(h5f_input['x_train']) / 5)):
            percent = i / int(len(h5f_input['x_train']) / 5) * 100
            # print('\r', percent, '% complited', end='')
            sys.stdout.write('\r%f complited' % percent)
            sys.stdout.flush()

            start_index = i * 5
            end_index = start_index + 5
            data = h5f_input['x_train'][start_index:end_index]
            prediction = model.predict_on_batch(data)
            if not np.allclose(prediction, h5f_ansers['x_train'][start_index:end_index, 1], atol=0.001):
                print('Index:')
                print(i * 5)
                print('Prediction:')
                print(prediction)
                print('Generated:')
                print(h5f_ansers['x_train'][start_index:end_index][1])
                raise ValueError('Wartości nie są identyczne')

        K.clear_session()

        print('Testing ground_truth:')
        model = load_model(path_to_weights)
        output = Lambda(CreateNN.soft_softmax_layer)(model.layers[-2].output)
        model = Model(inputs=model.inputs, outputs=output)
        optimizer = SGD(lr=0.1, momentum=0.9, nesterov=True)
        model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        for i in range(int(len(h5f_input['x_train']) / 5)):

            percent = i / int(len(h5f_input['x_train']) / 5) * 100
            # print('\r', percent, '% complited', end='')
            sys.stdout.write('\r%f complited' % percent)
            sys.stdout.flush()

            start_index = i * 5
            end_index = start_index + 5
            data = h5f_input['x_train'][start_index:end_index]
            prediction = model.predict_on_batch(data)
            if not np.allclose(prediction, h5f_ansers['x_train'][start_index:end_index, 0], atol=0.001):
                print('Index:')
                print(i * 5)
                print('Prediction:')
                print(prediction)
                print('Generated:')
                print(h5f_ansers['x_train'][start_index:end_index][0])
                raise ValueError('Wartości nie są identyczne')

        K.clear_session()