from GeneratorStorage.GeneratorDataLoader import GeneratorDataLoader
from TrainingData import TrainingData
import numpy as np


class GeneratorDataLoaderFromMemory(GeneratorDataLoader):

    def __init__(self, training_data: TrainingData, repeat_labels: int = 1):
        self.training_data = training_data
        self.repeat_labels = repeat_labels

    def get_generator_flow(self, data_generator, batch_size, shuffle, *kargs, **kwargs):
        generator = data_generator.flow(x=self.training_data.get_training_inputs(),
                                   y=self.training_data.get_training_outputs(),
                                   batch_size=batch_size,
                                   shuffle=shuffle)

        # generator.y = np.swapaxes(np.asarray([generator.y] * self.repeat_labels), 0, 1)
        generator.y = [generator.y] * self.repeat_labels
        return generator

