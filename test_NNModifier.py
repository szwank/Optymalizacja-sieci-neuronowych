import unittest
from testy1 import loss_layer, loss
import numpy as np


class TestLossLayer(unittest.TestCase):
    def test_computation(self):
        # test lomputation of loss layer
        x = np.ones(shape=(1, 10))
        y = np.ones(10)
        self.assertAlmostEqual(loss_layer(x), loss(y, y, y, y))
