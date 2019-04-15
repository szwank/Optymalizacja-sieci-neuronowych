import json
from keras.models import model_from_json
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import load_model
import os



class NNLoader:

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



        return [x_train, x_validation, x_test], [y_train, y_validation, y_test]

    @staticmethod
    def load_best_model_from_dir(directory):
        """Metoda wczytuje najleprzy model sieci neuronowej w folderze directory. Key_words, służą do filtrowania po
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

        return load_model(os.path.join(directory, list_of_files[position_of_best]))

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

