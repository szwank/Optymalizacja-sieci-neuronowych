import json
from keras.models import model_from_json
from keras.datasets import cifar10
from keras.utils import np_utils


class NNLoader:

    @staticmethod
    def load(file_name):
        with open(file_name) as json_file:
            model = json.load(json_file)

        return model_from_json(model)

    @staticmethod
    def load_CIFAR10():

        NUM_CLASSES = 10

        # Wczytanie bazy zdjęć
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # Wielkości zbiorów danych
        TRAIN_SIZE = int(0.9 * len(x_train))
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

