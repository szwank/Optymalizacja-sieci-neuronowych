import keras
import numpy as np


class Data_Generator_for_Shallowing(keras.utils.Sequence):
    def __init__(self, generator, repeat_correct_labels_x_times):
        self.generator = generator
        self.repeat_correct_labels_x_times = repeat_correct_labels_x_times

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.generator)

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        x, y_unit = self.generator[index]

        y = []
        for i in range(self.repeat_correct_labels_x_times):
            y.append(y_unit)

        # y.append(np.concatenate(y, axis=-1))

        return x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.generator.on_epoch_end()
