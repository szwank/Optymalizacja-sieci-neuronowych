import unittest
from keras.models import load_model, Model
from keras.optimizers import SGD
from keras.layers import Lambda
from CreateNN import CreateNN
import keras.backend as K
from NNLoader import NNLoader
import h5py
import numpy as np
from DataGenerator_for_knowledge_distillation import DataGenerator_for_knowledge_distillation
import sys


class TestDataGeneratorForKnowledgeDistillation(unittest.TestCase):
    def test_if_data_is_generated_correctly(self):
        path_to_weights = 'Zapis modelu/VGG16-CIFAR10-0.94acc.hdf5'
        model = load_model(path_to_weights)
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

        batch_size = 128
        [x_train, x_validation, x_test], [y_train, y_validation, y_test] = NNLoader.load_CIFAR10()
        params = {'dim': (32, 32, 3),
                  'batch_size': batch_size,
                  'number_of_classes': 10,
                  'shuffle': True}

        training_gen = DataGenerator_for_knowledge_distillation(name_of_data_set_in_file='x_train',
                                                                data_to_be_processed=x_train,
                                                                labels=y_train,
                                                                path_to_weights=path_to_weights,
                                                                **params)
        print('Testing logits:')
        optimizer = SGD(lr=0.1, momentum=0.9, nesterov=True)
        model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        for i in range(len(training_gen)):
            percent = i / len(training_gen) * 100
            # print('\r', percent, '% complited', end='')
            sys.stdout.write('\r%f complited' % percent)
            sys.stdout.flush()

            data = training_gen[i]
            prediction = model.predict_on_batch(data[0])
            if not np.allclose(prediction, data[1][:, 10:20], atol=0.001):
                print('Index:')
                print(i * batch_size)
                print('Prediction:')
                print(prediction)
                print('Generated:')
                print(data[1][:, 10:20])
                raise ValueError('Wartości nie są identyczne')

        K.clear_session()

