import numpy as np
import sys
from keras.models import Model, load_model
from DataGenerator import DataGenerator
import keras.backend as K
import tensorflow as tf
import gc



class DataGenerator_for_knowledge_distillation(DataGenerator):

    def __init__(self, generator, path_to_weights, shuffle=True):
        """Initialization"""
        self.LABELS = 0
        self.LOGITS = 1

        self.dim = generator.x.shape[1:]
        self.batch_size = generator.batch_size
        self.number_of_classes = generator.y.shape[-1]
        self.number_of_data = generator.n

        self.input_data, self.correct_labels = self.__generate_inputs_and_correct_labels(generator)

        self.indexes = np.arange(self.number_of_data)

        self.path_to_weights = path_to_weights
        self.neural_network = self.convert_original_neural_network(load_model(path_to_weights))

        self.predictions = self.__Generate_predictions()

        self.shuffle = shuffle
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
        inputs = inputs

        return inputs, answers


    def __Generate_predictions(self):
        dimensions_of_dataset = (self.number_of_data, 2, self.number_of_classes)

        generated_data = np.zeros(dimensions_of_dataset)
        print('Generator starting generating predictions of original network\n')

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

            sys.stdout.write('\r%f complited' % percent)
            sys.stdout.flush()

        print('\nGeneration completed\n')

        return generated_data

    def __process_batch_of_data(self, data):

        # data = np.expand_dims(data, axis=0)     # Kłopoty z formatem
        procesed_data = self.neural_network.predict_on_batch(data)  # Wygenerowanie i
        # utworzenie danych
        procesed_data = np.asarray(procesed_data)  # Sformatowanie danych
        procesed_data = np.swapaxes(procesed_data, 0, 1)  # format wyjściowy(batch size, ilość wyjść sieci po
        # przerobieniu, wymiar wyjść)
        return procesed_data


    def convert_original_neural_network(self, model):
        return Model(inputs=model.inputs, outputs=(model.layers[-1].output, model.layers[-2].output))

    def __load_weights_to_neural_network(self):
        self.neural_network.load_weights(self.path_to_weights, by_name=True)

    def __generate_inputs_and_correct_labels(self, generator):
        input_dimension = (self.number_of_data, *self.dim)
        input = np.empty(input_dimension)
        correct_label_dimension = (self.number_of_data, self.number_of_classes)
        correct_labels = np.empty(correct_label_dimension)

        for i in range(len(generator)):
            data = generator[i]
            start_index = i * self.batch_size
            end_index = start_index + len(data[1])
            input[start_index:end_index] = data[0]
            correct_labels[start_index:end_index] = data[1]
        return input, correct_labels

    def reset_keras_session(self):
        sess = K.get_session()
        K.clear_session()
        sess.close()

        print(gc.collect())  # if it's done something you should see a number being outputted

        # use the same config as you used to create the session
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = "0"
        # config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 1

