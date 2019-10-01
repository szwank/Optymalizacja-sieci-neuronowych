import unittest

from keras import Input, Model, backend as K
from keras.layers import Lambda, Softmax
from CreateNN import CreateNN
import json
import numpy as np

from CreateNN import CreateNN
from NNModifier import NNModifier
from keras import backend as K



class TestLossLayer(unittest.TestCase):
    def test_loos_layer_positive_numbers(self):
        # test lomputation of loss layer
        x = np.ones(shape=(1, 10))
        y = np.ones(10)

        for i in np.arange(0, 10, 0.3):
            print(i)
            loss_layer_input = [i * x, i * x, i * x, i * x]
            self.assertAlmostEqual(TestLossLayer.loss_layer(loss_layer_input), TestLossLayer.loss(i*y, i*y, i*y, i*y), 2)

    def test_loos_layer_negative_logits(self):
        x = np.ones(shape=(1, 10))
        y = np.ones(10)

        for i in np.arange(-10, 0, 0.3):
            print(i)
            loss_layer_input = [-i * x, i * x, -i * x, i * x]
            self.assertAlmostEqual(self.loss_layer(loss_layer_input), self.loss(-i * y, i * y, -i * y, i * y), 2)

    @staticmethod
    def loss(ground_truth, logits, ground_truth_student, logits_student):
        T = 300
        alfa = 0.05
        first_part = -np.sum(np.multiply(ground_truth, np.log(ground_truth_student + K.epsilon())))
        q = np.divide(np.exp(np.divide(logits_student, T)), np.sum(np.exp(np.divide(logits_student, T))))
        p = np.divide(np.exp(np.divide(logits, T)), np.sum(np.exp(np.divide(logits, T)),))
        second_part = - alfa * np.sum(np.multiply(p, np.log(q + K.epsilon())))

        return first_part

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