import unittest
from GeneratorStorage.GeneratorDataLoader import GeneratorDataLoader
from unittest.mock import Mock
import numpy as np

class MockImageDataGenerator(Mock):
    x = np.ones((1, 32, 32, 3))
    y = np.ones((1, 10))

    def flow(self, *args):
        yield [self.x, self.y]

class TestDataLoader(unittest.TestCase):

    def test_init(self):
        with self.assertRaises(NotImplementedError):
            GeneratorDataLoader()

    def test_get_generator_flow(self):
        mock_image_data_generator = MockImageDataGenerator

        class GeneratorDataLoaderTester(GeneratorDataLoader):
            def __init__(self):
                pass

        with self.assertRaises(NotImplementedError):


            object = GeneratorDataLoaderTester()
            object.get_generator_flow(data_generator=mock_image_data_generator, batch_size=1, shuffle=True,
                                      type_of_generator='train')

class TestDataLoaderFromMemory(unittest.TestCase):

    def test_init