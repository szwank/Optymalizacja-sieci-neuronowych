from GeneratorStorage.GeneratorDataLoader import GeneratorDataLoader
from TrainingData import TrainingData
import numpy as np


class GeneratorDataLoaderFromMemory(GeneratorDataLoader):

    def __init__(self, training_data: TrainingData):
        self.training_data = training_data

    def get_generator_flow(self, data_generator, batch_size, shuffle, *kargs, **kwargs):
        return data_generator.flow(x=self.training_data.get_training_inputs(),
                                   y=self.training_data.get_training_outputs(),
                                   batch_size=batch_size,
                                   shuffle=shuffle)


