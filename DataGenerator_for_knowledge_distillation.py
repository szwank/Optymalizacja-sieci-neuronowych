from keras.utils import Sequence
import numpy as np


class DataGenerator_for_knowledge_distillation(Sequence):
    def __init__(self, generator, number_of_repetitions_of_input_data, repeat_correct_labels_x_times):
        self.generator = generator
        self.repeat_correct_labels_x_times = repeat_correct_labels_x_times
        self.number_of_repetitions_of_input_data = number_of_repetitions_of_input_data

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.generator)

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        x_unit, y_unit = self.generator[index]

        # x = np.tile(x_unit, self.number_of_repetitions_of_input_data)
        y = np.tile(y_unit, (self.repeat_correct_labels_x_times, 1))
        y = np.swapaxes(y, 0, 1)

        # y = []
        # for i in range(self.repeat_correct_labels_x_times):
        #     y.append(y_unit)

        x = []
        for i in range(self.number_of_repetitions_of_input_data):
            x.append(x_unit)

        return x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.generator.on_epoch_end()

