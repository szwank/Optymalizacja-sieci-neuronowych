import numpy as np
import keras
import h5py



class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x_data_name, y_data_name, data_dir, batch_size=32, dim=(32, 32), n_channels=3,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        # self.labels = labels
        # self.list_IDs = list_IDs
        self.h5_file_to_be_processed = h5py.File(data_dir, 'r')
        self.number_of_channels_in_data = n_channels
        self.number_of_classes = n_classes
        self.shuffle = shuffle
        self.name_of_dataset_in_file = x_data_name
        self.y_data_name = y_data_name
        self.indexes = np.arange(len(self.h5_file_to_be_processed[x_data_name]))
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'

        return int(np.floor(len(self.h5_file_to_be_processed[self.name_of_dataset_in_file]) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        # indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(index)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.number_of_channels_in_data))
        y = np.empty((self.batch_size, self.number_of_classes), dtype=int)

        # Generate data
        for i, ID in enumerate(indexes):
            # Store sample
            # X[i,] = np.load('data/' + ID + '.npy')
            X[i, ] = self.h5_file_to_be_processed[self.name_of_dataset_in_file][ID]
            # Store class
            y[i, ] = self.h5_file_to_be_processed[self.y_data_name][ID]

        return X, y
