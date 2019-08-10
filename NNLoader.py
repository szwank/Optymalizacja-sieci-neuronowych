import json
from keras.models import model_from_json
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import load_model
from DataStorage import DataStorage
from TrainingData import TrainingData
import os
import re
import time


class NNLoader:
    def try_load_model(path_to_model: str, compile: bool = True, number_of_trials: int = 10):
        try:
            return load_model(path_to_model)
        except:
            if number_of_trials > 0:
                time.sleep(0.5)
                number_of_trials -= 1
                return NNLoader.try_load_model(path_to_model, compile, number_of_trials)
            else:
                raise ValueError("Cannot open model in path {}. The number of attempts to load model has been exceeded."
                                 " Check if path is correct.".format(path_to_model))


    @staticmethod
    def load(file_name):
        with open(file_name) as json_file:
            model = json.load(json_file)

        return model_from_json(model)

    @staticmethod
    def load_CIFAR10(validation_split=0.9):

        NUM_CLASSES = 10

        # Wczytanie bazy zdjęć
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # Wielkości zbiorów danych
        TRAIN_SIZE = int(validation_split * len(x_train))
        VALIDATION_SIZE = int(len(x_train) - TRAIN_SIZE)

        # podział zbioru treningowego na treningowy i walidacyjny
        x_validation = x_train[:VALIDATION_SIZE]
        x_train = x_train[VALIDATION_SIZE:]
        y_validation = y_train[:VALIDATION_SIZE]
        y_train = y_train[VALIDATION_SIZE:]

        # Zamiana numeru odpowiedzi na macierz labeli
        y_train = np_utils.to_categorical(y_train, NUM_CLASSES)
        y_validation = np_utils.to_categorical(y_validation, NUM_CLASSES)
        y_test = np_utils.to_categorical(y_test, NUM_CLASSES)

        input_data = DataStorage(x_train, x_validation, x_test)
        output_labels = DataStorage(y_train, y_validation, y_test)
        training_data = TrainingData(input_data, output_labels)

        return training_data

    @staticmethod
    def load_best_model_from_dir(directory, mode):
        """Metoda wczytuje najleprzy model sieci neuronowej w folderze directory. argument mode słuzy do stwierdzenia
        czy szukamy najmniejszego czy największego parametru w nazwie.
        mode, możliwe parametry: 'lowest', 'highest'."""

        list_of_files_names = os.listdir(directory)
        position_of_best = 0
        best_evaluation_parameter = re.findall("\d+\.\d+", list_of_files_names[0])
        
        if mode not in ['lowest', 'highest']:
            raise ValueError('Nie znana wartośc parametru mode. Dostępne: "lowest", "highest"')

        for i, file_name in enumerate(list_of_files_names):
            if 'weights-improvement' in file_name:
                evaluation_parameter = re.findall("\d+\.\d+", file_name)
                if mode is 'highest':
                    if evaluation_parameter > best_evaluation_parameter:
                        best_evaluation_parameter = evaluation_parameter
                        position_of_best = i

                if mode is 'lowest':
                    if evaluation_parameter < best_evaluation_parameter:
                        best_evaluation_parameter = evaluation_parameter
                        position_of_best = i

        return NNLoader.try_load_model(os.path.join(directory, list_of_files_names[position_of_best]), compile=False)

    @staticmethod
    def load_best_weights_from_dir(model, directory):
        """Metoda wczytuje najleprze wagi sieci neuronowej w folderze directory. Key_words, służą do filtrowania po
        nazwie. Wczytywany model musi posiadać słowa kluczowe."""

        list_of_files = os.listdir(directory)
        position_of_best = 0
        best_accuracy = 0

        for i, file in enumerate(list_of_files):
            if 'weights-improvement' in file:
                split_string = file.split('.')
                accuracy = float(split_string[1])
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    position_of_best = i

        return model.load_weights(os.path.join(directory, list_of_files[position_of_best]))

    @staticmethod
    def load_weights_from_list(model, weights, debug=False):
        if debug is True:
            print('\nLoading weights to the network:')

        actual_weight_index = 0

        for layer in model.layers:
            weights_from_model = layer.get_weights()
            if debug is True:
                print('Loading weights to {} layer'.format(layer.name))
            positions_to_take = len(weights_from_model)

            if positions_to_take > 0:
                start_index = actual_weight_index
                end_index = start_index + positions_to_take
                weights_for_layer = weights[start_index:end_index]
                layer.set_weights(weights_for_layer)
                actual_weight_index = end_index

            if actual_weight_index >= len(weights):
                break

        return model


