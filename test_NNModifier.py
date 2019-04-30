import unittest

from keras import Input, Model, backend as K
from keras.layers import Lambda, Softmax

from CreateNN import CreateNN
import numpy as np


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

    def test_loos_layer_biger_numbers(self):
        x = np.ones(shape=(1, 10))
        y = np.ones(10)

        for i in [199.88783, 1034, 4236.883218, 7990]:
            print(i)
            loss_layer_input = [i * x, i * x, i * x, i * x]
            self.assertAlmostEqual(self.loss_layer(loss_layer_input), self.loss(i * y, i * y, i * y, i * y), 2)

    @staticmethod
    def loss_layer(x):

        ground_truth = Input(shape=(10,), name='ground_truth')  # Dodatkowej wejście na wyjścia z orginalnej sieci(SoftMax)
        logits = Input(shape=(10,), name='logits')  # Dodatkowe wejście na wyjścia z orginalnej sieci(warstwa przed SoftMax)
        ground_truth_student = Input(shape=(10,), name='ground_truth_student')  # Dodatkowej wejście na wyjścia z orginalnej sieci(SoftMax)
        logits_student = Input(shape=(10,), name='logits_student')  # Dodatkowe wejście na wyjścia z orginalnej sieci(warstwa przed SoftMax)

        # Warstwa obliczająca loss dla procesu knowledge distillation
        output = Lambda(CreateNN.loss_for_knowledge_distillation_layer, name='loss')([ground_truth, logits,
                                                                                    ground_truth_student,
                                                                                    logits_student])

        model = Model(inputs=(ground_truth, logits, ground_truth_student, logits_student), outputs=output)

        prediction = model.predict(x=x)
        K.clear_session()
        return float(prediction)

    def test_loos_method(self):
        pass

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
            if not np.allclose(model_tested.predict(x, batch_size=2), model.predict(x, batch_size=2)):
                print('Data:')
                print(x)
                print('Tested:')
                print(model_tested.predict(x, batch_size=2))
                print('Correct:')
                print(model.predict(x, batch_size=2))
                raise ValueError('Wartości nie są identyczne')


