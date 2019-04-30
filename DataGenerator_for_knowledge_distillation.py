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
        self.h5_file_to_be_processed = h5py.File(path_to_h5py_data_to_be_processed, 'r')
        self.number_of_classes = number_of_classes
        self.shuffle = shuffle
        self.name_of_dataset_in_file = name_of_data_set_in_file
        self.number_of_data = len(self.h5_file_to_be_processed[name_of_data_set_in_file])
        self.indexes = np.arange(self.number_of_data)
        self.path_to_weights = path_to_weights
        self.neural_network = self.convert_original_neural_network(load_model(path_to_weights))
        self.number_of_inputs_in_trained_network = inputs_number

        self.h5_file_predictions = self.__Generate_predictions()
        self.on_epoch_end()

    def _DataGenerator__data_generation(self, indexes):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization
        y = np.zeros((self.batch_size), dtype=int)
        x = []
        x.append(np.empty((self.batch_size, self.number_of_classes)))
        x.append(np.empty((self.batch_size, self.number_of_classes)))
        x.append(np.empty((self.batch_size, *self.dim)))

        # Generate data
        for i, ID in enumerate(indexes):
            # Store sample
            # X[i,] = np.load('data/' + ID + '.npy')
            x[0][i], x[1][i] = np.split(self.h5_file_predictions[self.name_of_dataset_in_file][ID], 2, axis=0)
            x[2][i] = self.h5_file_to_be_processed[self.name_of_dataset_in_file][ID]
            # np.split(X, np.arange(self.inputs_number), axis=1 )[1:4]

        return x[2], [x[0], x[1]]

    def __Generate_predictions(self):
        if not os.path.exists('temp/'):  # Stworzenie folderu jeżeli nie istnieje.
            os.makedirs('temp/')

        h5f = h5py.File('temp/Generator_data.h5', 'a')

        if self.name_of_dataset_in_file in list(h5f.keys()):  # sprawdzenie czy taki dataset istnieje
            del h5f[
                self.name_of_dataset_in_file]  # jeżeli tak usunięcie. Brak usunięcia spowoduje błędy w przypadku istnienia.

        dimensions_of_dataset = (
            self.number_of_data, self.number_of_inputs_in_trained_network - 1,
            self.number_of_classes)
        dataset_to_save_generated_data = h5f.create_dataset(self.name_of_dataset_in_file, dimensions_of_dataset)

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

            data = self.h5_file_to_be_processed[self.name_of_dataset_in_file][start_index:end_index]  # Pobranie danych

            processed_data = self.__process_batch_of_data(data)

            dataset_to_save_generated_data[start_index:end_index, ] = processed_data

            percent = i * self.batch_size / self.number_of_data * 100
            # print('\r', percent, '% complited', end='')
            sys.stdout.write('\r%f complited' % percent)
            sys.stdout.flush()

        print('\nGeneration completed')

        return h5f

    def __process_batch_of_data(self, data):

        # data = np.expand_dims(data, axis=0)     # Kłopoty z formatem
        procesed_data = self.neural_network.predict_on_batch(data)  # Wygenerowanie i
        # utworzenie danych
        procesed_data = np.asarray(procesed_data)  # Sformatowanie danych
        procesed_data = np.swapaxes(procesed_data, 0, 1)  # format wyjściowy(batch size, ilość wyjść sieci po
        # przerobieniu, wymiar wyjść)
        return procesed_data

    # def __del__(self):
    #     if os.path.isfile('temp/Generator_data.h5'):  # Sprawdzenie czy istieje plik wygenerowany przez generator
    #         os.remove('temp/Generator_data.h5')     # Usunięcie jeżeli taki istnieje

    def convert_original_neural_network(self, model):
        output = Lambda(CreateNN.soft_softmax_layer(T=15))(model.layers[-2].output)
        return Model(inputs=model.inputs, outputs=(output, model.layers[-2].output))

    def __load_weights_to_neural_network(self):
        self.neural_network.load_weights(self.path_to_weights, by_name=True)
