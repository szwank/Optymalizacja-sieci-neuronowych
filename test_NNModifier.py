import unittest

from keras import Input, Model
from keras.layers import Lambda, Softmax

import json
import numpy as np

from CreateNN import CreateNN
from NNModifier import NNModifier
from keras import backend as K

class TestSoftSoftMax(unittest.TestCase):
    def test_calculation(self):
        inputs = Input(shape=(10,))
        output = Lambda(CreateNN.soft_softmax_layer)(inputs)
        model_tested = Model(inputs=inputs, outputs=output)

        inputs = Input(shape=(10,))
        output = Softmax()(inputs)
        model = Model(inputs=inputs, outputs=output)

        for i in range(100):
            x = 100 * np.random.randn(2, 10)
            self.assertTrue(np.allclose(model_tested.predict(x, batch_size=2), model.predict(x, batch_size=2)),
                            'Data: {}\n Tested: {}\n Correct: {}'.format(x, model_tested.predict(x, batch_size=2), model.predict(x, batch_size=2)))


class TestRemoveChosenFiltersFromModel(unittest.TestCase):

    def remove_filters_and_test(self, number_of_filters_to_remove: list):
        model = CreateNN.create_VGG16_for_CIFAR10()
        number_of_filters_in_network = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]

        filters_to_remove = {}
        for i, element in enumerate(number_of_filters_to_remove):
                filters_to_remove.update({i+1: np.arange(1, element+1)})

        model = NNModifier.remove_chosen_filters_from_model(model, filters_to_remove, 1, debug=True)
        model_dictionary = json.loads(model.to_json())

        conv_layer_counter = 0
        for layer in model_dictionary['config']['layers']:
            if 'conv' in layer['name']:
                conv_layer_counter += 1
                number_of_filters_in_actual_conv_layer = number_of_filters_in_network[conv_layer_counter-1] -\
                                                         number_of_filters_to_remove[conv_layer_counter-1]
                self.assertEqual(layer['config']['filters'], number_of_filters_in_actual_conv_layer)

    def test_removing_filters(self):

        number_of_filters_to_remove = [63, 63, 127, 127, 255, 255, 255, 511, 511, 511, 511, 511, 511]
        self.remove_filters_and_test(number_of_filters_to_remove)
        K.clear_session()

        number_of_filters_to_remove = [63, 0, 0, 0, 255, 255, 0, 0, 511, 0, 0, 511, 511]
        self.remove_filters_and_test(number_of_filters_to_remove)
        K.clear_session()

        number_of_filters_to_remove = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.remove_filters_and_test(number_of_filters_to_remove)
        K.clear_session()

        number_of_filters_to_remove = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.remove_filters_and_test(number_of_filters_to_remove)
        K.clear_session()