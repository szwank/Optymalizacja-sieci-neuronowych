import numpy as np
import h5py
import os
import sys
from keras.models import Model, load_model
from keras.layers import Lambda
from DataGenerator import DataGenerator
from CreateNN import CreateNN
from Create_NN_graph import Create_NN_graph


class DG_for_kd(DataGenerator):

    def __init__(self, x_data_name, data_dir, dir_to_weights, batch_size=32, dim=(32, 32), n_channels=3,
                 n_classes=10, shuffle=True, inputs_number=3):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        # self.labels = labels
        # self.list_IDs = list_IDs
        self.h5_file = h5py.File(data_dir, 'r')
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.x_data_name = x_data_name
        self.indexes = np.arange(len(self.h5_file[x_data_name]))
        self.original_network = self.convert_original_neural_network(load_model(dir_to_weights))
        self.inputs_number = inputs_number

        self.h5_file_predictions = self.__Generate_predictions()
        self.on_epoch_end()

    def _DataGenerator__data_generation(self, indexes):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization
        y = np.zeros((self.batch_size), dtype=int)
        x = []
        x.append(np.empty((self.batch_size, self.n_classes)))
        x.append(np.empty((self.batch_size, self.n_classes)))
        x.append(np.empty((self.batch_size, *self.dim, self.n_channels)))

        # Generate data
        for i, ID in enumerate(indexes):
            # Store sample
            # X[i,] = np.load('data/' + ID + '.npy')
            x[0][i], x[1][i] = np.split(self.h5_file_predictions[self.x_data_name][ID], 2, axis=0)
            x[2][i] = self.h5_file[self.x_data_name][ID]
            # np.split(X, np.arange(self.inputs_number), axis=1 )[1:4]

        return x, y

    def __Generate_predictions(self):
        if not os.path.exists('temp/'):  # Stworzenie folderu jeżeli nie istnieje.
            os.makedirs('temp/')

        number_of_data = len(self.h5_file[self.x_data_name])

        h5f = h5py.File('temp/Generator_data.h5', 'a')

        if self.x_data_name in list(h5f.keys()):    # sprawdzenie czy taki dataset istnieje
            del h5f[self.x_data_name]   # jeżeli tak usunięcie. Brak usunięcia spowoduje błędy w przypadku istnienia.
        data_set = h5f.create_dataset(self.x_data_name, (number_of_data, self.inputs_number-1, self.n_classes))   # Stworzenie pustej klasy w pliku

        print('Generator starting generating predictions of original network for', self.x_data_name, '\n')

        for i in range(int(np.ceil(number_of_data/self.batch_size))):     # Generowanie danych w partiach

            if (i+1) * self.batch_size < number_of_data:
                generate_batch_size = self.batch_size   # Wielkość partij gdy pozostało wystarczająco dużo danych
            else:
                generate_batch_size = number_of_data - (i * self.batch_size)  # Wielkość partij gdy ilość
                                                                    # pozostałych danych jest mniejsza niż batch_size

            data = self.h5_file[self.x_data_name][i:(i + generate_batch_size)]  # Pobranie danych
            # data = np.expand_dims(data, axis=0)     # Kłopoty z formatem
            data = self.original_network.predict_on_batch(data)  # Wygenerowanie i
                                                                                                  # utworzenie danych
            data = np.asarray(data)     # Sformatowanie danych
            data = np.swapaxes(data, 0, 1)      # format wyjściowy(batch size, ilość wyjść orginalnej sieci po
                                                # przerobieniu, wymiar wyjść)

            data_set[i*self.batch_size:(i*self.batch_size + generate_batch_size), ] = data

            percent = i * self.batch_size / number_of_data * 100
            # print('\r', percent, '% complited', end='')
            sys.stdout.write('\r%f complited' % percent)
            sys.stdout.flush()

        print('\nGeneration completed')

        return h5f

    # def __del__(self):
    #     if os.path.isfile('temp/Generator_data.h5'):  # Sprawdzenie czy istieje plik wygenerowany przez generator
    #         os.remove('temp/Generator_data.h5')     # Usunięcie jeżeli taki istnieje

    def convert_original_neural_network(self, model):
        Create_NN_graph.create_NN_graph(Model(inputs=model.inputs, outputs=(model.layers[-1].output, model.layers[-2].output)), name='generator_model')
        output = Lambda(CreateNN.soft_softmax_layer)(model.layers[-2].output)
        return Model(inputs=model.inputs, outputs=(output, model.layers[-2].output))


