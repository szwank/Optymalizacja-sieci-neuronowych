import numpy as np
import h5py
import os
import sys
from keras.models import Model, load_model
from keras.layers import Lambda
from DataGenerator import DataGenerator
from CreateNN import CreateNN
from Create_NN_graph import Create_NN_graph


class DataGenerator_for_knowledge_distillation(DataGenerator):

    def __init__(self, name_of_data_set_in_file, path_to_h5py_data_to_be_processed, path_to_weights, batch_size=32,
                 dim=(32, 32, 3), number_of_classes=10, shuffle=True, inputs_number=3):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        # self.labels = labels
        # self.list_IDs = list_IDs
        self.path_to_h5py_data_to_be_processed = path_to_h5py_data_to_be_processed

        self.number_of_classes = number_of_classes
        self.shuffle = shuffle
        self.name_of_dataset_in_file = name_of_data_set_in_file

        self.number_of_data = self.__get_number_of_file_to_be_pocesed()
        self.indexes = np.arange(self.number_of_data)
        self.path_to_weights = path_to_weights
        self.neural_network = self.convert_original_neural_network(load_model(path_to_weights))
        self.number_of_inputs_in_trained_network = inputs_number

        self.path_to_generated_file = 'temp/Generator_data.h5'

        if not self.__check_if_correct_data_exist():
            self.h5_file_predictions = self.__Generate_predictions()
        else:
            self.h5_file_predictions = h5py.File('temp/Generator_data.h5', 'a')
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(self.number_of_data/self.batch_size)

    def _DataGenerator__data_generation(self, indexes):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)

        indexes.sort()
        indexes = np.ndarray.tolist(indexes)

        h5_file_to_be_processed = h5py.File(self.path_to_h5py_data_to_be_processed, 'r')

        ansers = self.h5_file_predictions[self.name_of_dataset_in_file][indexes]
        ansers = np.reshape(ansers, (self.batch_size, self.number_of_classes*2))
        input = h5_file_to_be_processed[self.name_of_dataset_in_file][indexes]
        input = input / 255.0
        # ansers = np.concatenate((softmax_output, before_softmax_output), axis=-1)
        h5_file_to_be_processed.close()
        return input, ansers

    def __Generate_predictions(self):
        if not os.path.exists('temp/'):  # Stworzenie folderu jeżeli nie istnieje.
            os.makedirs('temp/')

        h5f = h5py.File(self.path_to_generated_file, 'a')

        if self.name_of_dataset_in_file in list(h5f.keys()):  # sprawdzenie czy taki dataset istnieje
            del h5f[
                self.name_of_dataset_in_file]  # jeżeli tak usunięcie. Brak usunięcia spowoduje błędy w przypadku istnienia.

        dimensions_of_dataset = (
            self.number_of_data, self.number_of_inputs_in_trained_network - 1,
            self.number_of_classes)
        dataset_to_save_generated_data = h5f.create_dataset(self.name_of_dataset_in_file, dimensions_of_dataset, dtype='float64')

        print('Generator starting generating predictions of original network for', self.name_of_dataset_in_file, '\n')

        self.__load_weights_to_neural_network()

        h5_file_to_be_processed = h5py.File(self.path_to_h5py_data_to_be_processed, 'r')

        number_of_batches = int(np.ceil(self.number_of_data / self.batch_size))

        for i in range(number_of_batches):  # Generowanie danych w partiach

            if (i + 1) * self.batch_size < self.number_of_data:
                generated_batch_size = self.batch_size  # Wielkość partij gdy pozostało wystarczająco dużo danych
            else:
                generated_batch_size = self.number_of_data - (i * self.batch_size)  # Wielkość partii gdy ilość
                # pozostałych danych jest mniejsza niż batch_size

            start_index = i * self.batch_size
            end_index = start_index + generated_batch_size

            data = h5_file_to_be_processed[self.name_of_dataset_in_file][start_index:end_index]  # Pobranie danych

            processed_data = self.__process_batch_of_data(data)

            dataset_to_save_generated_data[start_index:end_index, ] = processed_data

            percent = i * self.batch_size / self.number_of_data * 100
            # print('\r', percent, '% complited', end='')
            sys.stdout.write('\r%f complited' % percent)
            sys.stdout.flush()

        h5_file_to_be_processed.close()
        print('\nGeneration completed')

        return h5f

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

    def __check_if_correct_data_exist(self):
        if not os.path.exists(self.path_to_generated_file):
            return False

        h5f = h5py.File(self.path_to_generated_file, 'r')
        if self.name_of_dataset_in_file not in h5f.keys():
            h5f.close()
            return False

        h5_file_to_be_processed = h5py.File(self.path_to_h5py_data_to_be_processed, 'r')
        indexes = np.arange(start=0, stop=self.batch_size)
        indexes = np.ndarray.tolist(indexes)
        data = h5_file_to_be_processed[self.name_of_dataset_in_file][indexes]
        proceded_data = self.__process_batch_of_data(data)


        data_from_file = h5f[self.name_of_dataset_in_file][indexes]
        h5f.close()
        h5_file_to_be_processed.close()
        if np.allclose(proceded_data, data_from_file, atol=0.0001):
            print('Data was generated before. Skipping data generation.')
            return True
        else:
            return False

    def __get_number_of_file_to_be_pocesed(self):
        h5f = h5py.File(self.path_to_h5py_data_to_be_processed, 'r')
        number = len(h5f[self.name_of_dataset_in_file])
        h5f.close()
        return number
