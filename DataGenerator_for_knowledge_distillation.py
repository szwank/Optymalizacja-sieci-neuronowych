import numpy as np
import h5py
import os
import sys
from keras.models import Model, load_model
from keras.layers import Lambda
from DataGenerator import DataGenerator
from CreateNN import CreateNN
from Create_NN_graph import Create_NN_graph
from mpi4py import MPI


class DataGenerator_for_knowledge_distillation(DataGenerator):

    def __init__(self, name_of_data_set_in_file, data_to_be_processed, labels, path_to_weights, batch_size=32,
                 dim=(32, 32, 3), number_of_classes=10, shuffle=True, path_to_generated_file='temp/Generator_data.h5'):
        """Initialization"""
        self.LABELS = 0
        self.LOGITS = 1

        self.dim = dim
        self.batch_size = batch_size

        self.number_of_classes = number_of_classes
        self.shuffle = shuffle
        self.name_of_dataset_in_file = name_of_data_set_in_file

        self.input_data = data_to_be_processed
        self.correct_labels = labels

        self.number_of_data = self.__get_number_of_data_to_be_pocesed()
        self.indexes = np.arange(self.number_of_data)
        self.path_to_weights = path_to_weights
        self.neural_network = self.convert_original_neural_network(load_model(path_to_weights))
        self.path_to_generated_file = path_to_generated_file

        self.predictions = self.__Generate_predictions()

        self.on_epoch_end()  # Musi być wywołana ostatnia

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(self.number_of_data/self.batch_size)

    def _DataGenerator__data_generation(self, indexes):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)

        indexes.sort()
        indexes = np.ndarray.tolist(indexes)

        correct_labels = self.correct_labels[indexes]
        predicted_logits = self.predictions[indexes, self.LOGITS]
        answers = np.concatenate((correct_labels, predicted_logits), axis=-1)

        inputs = self.input_data[indexes]
        inputs = inputs / 255.0

        return inputs, answers


    def __Generate_predictions(self):
        dimensions_of_dataset = (self.number_of_data, 2, self.number_of_classes)

        generated_data = np.zeros(dimensions_of_dataset)
        print('Generator starting generating predictions of original network for', self.name_of_dataset_in_file, '\n')

        self.__load_weights_to_neural_network()

        number_of_batches = int(np.ceil(self.number_of_data / self.batch_size))

        for i in range(number_of_batches):  # Generowanie danych w partiach

            if (i + 1) * self.batch_size < self.number_of_data:
                generated_batch_size = self.batch_size  # Wielkość partij gdy pozostało wystarczająco dużo danych
            else:
                generated_batch_size = self.number_of_data - (i * self.batch_size)  # Wielkość partii gdy ilość
                # pozostałych danych jest mniejsza niż batch_size

            start_index = i * self.batch_size
            end_index = start_index + generated_batch_size

            data = self.input_data[start_index:end_index]  # Pobranie danych

            processed_data = self.__process_batch_of_data(data)

            generated_data[start_index:end_index, ] = processed_data

            percent = i * self.batch_size / self.number_of_data * 100
            # print('\r', percent, '% complited', end='')
            sys.stdout.write('\r%f complited' % percent)
            sys.stdout.flush()

        print('\nGeneration completed')

        return generated_data

    def __process_batch_of_data(self, data):

        # data = np.expand_dims(data, axis=0)     # Kłopoty z formatem
        procesed_data = self.neural_network.predict_on_batch(data/255.0)  # Wygenerowanie i
        # utworzenie danych
        procesed_data = np.asarray(procesed_data)  # Sformatowanie danych
        procesed_data = np.swapaxes(procesed_data, 0, 1)  # format wyjściowy(batch size, ilość wyjść sieci po
        # przerobieniu, wymiar wyjść)
        return procesed_data

    # def __del__(self):
    #     if os.path.isfile('temp/Generator_data.h5'):  # Sprawdzenie czy istieje plik wygenerowany przez generator
    #         os.remove('temp/Generator_data.h5')     # Usunięcie jeżeli taki istnieje

    def convert_original_neural_network(self, model):
        return Model(inputs=model.inputs, outputs=(model.layers[-1].output, model.layers[-2].output))

    def __load_weights_to_neural_network(self):
        self.neural_network.load_weights(self.path_to_weights, by_name=True)

    def __get_number_of_data_to_be_pocesed(self):
        number = len(self.input_data)
        return number



