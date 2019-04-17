import tensorflow as tf
import keras
from keras.applications.vgg16 import VGG16
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, MaxPool2D, Conv2D, Flatten, Dropout, Activation
from keras.models import Model, model_from_json, load_model
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras.layers import Softmax
import keras.backend as K
import numpy as np
from keras.utils import plot_model
import datetime
import os
from NNModifier import NNModifier
from NNLoader import NNLoader
from CreateNN import CreateNN
from Create_NN_graph import Create_NN_graph
from DataGenerator_for_knowledge_distillation import DG_for_kd
import json

def check_weights_was_changed(old_model, new_model):
    old_weights = old_model.get_weights()
    new_weights = new_model.get_weights()

    for i, (old_weight, new_weight) in enumerate(zip(old_weights, new_weights)):
        if np.array_equal(old_weight, new_weight):
            print(i, ': takie same')
        else:
            print(i, ': inne')

    for i, (l1, l2) in enumerate(zip(old_model.layers, new_model.layers)):
        print(i, l1.get_config() == l2.get_config())

def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)

def set_weights_as_ones(model):
    weights = model.get_weights()
    for i, element in enumerate(weights):
        dimension = element.shape
        weights[i] = np.ones(dimension)
    model.set_weights(weights)


def loss_for_knowledge_distillation(y_true, y_pred):
    return y_pred

def add_score_to_file(score, file_name='Skutecznosc warstw.json'):
    """Dopisanie wyniku klasyfikatora do pliku tekstowego."""

    conv_layer_number = score[2]
    loss = score[0]
    accuracy = score[1]

    if os.path.exists(file_name):
        file = open(file_name, "r")
        json_string = file.read()
        dictionary = json.loads(json_string)
        subordinate_dictionary = {str(conv_layer_number): {'loss': loss, 'accuracy': accuracy}}
        dictionary.update(subordinate_dictionary)
        file.close()
    else:
        dictionary = {str(conv_layer_number): {'loss': loss, 'accuracy': accuracy}}

    file = open(file_name, "w")
    json_string = json.dumps(dictionary)
    file.write(json_string)
    file.close()

def assesing_conv_layers(path_to_model, start_from_layer= 1, BATCH_SIZE=256):
    """Metoda oceniająca skuteczność poszczegulnych warstw konwolucyjnych"""
    model = load_model(path_to_model)
    model.summary()

    model_architecture = model.to_json(indent=4)
    model_architecture = json.loads(model_architecture)

    del(model)

    count_conv_layer = 0      # Licznik warstw konwolucyjnych.
    number_of_layers_in_model = len(model_architecture["config"]["layers"])

    for i in range(number_of_layers_in_model):

        if model_architecture["config"]["layers"][i]["class_name"] == 'Conv2D':     # Sprawdzenie czy dana warstwa jest konwolucyjna
            count_conv_layer += 1     # Zwiekszenie licznika

            if start_from_layer <= i:
                print('Testowanie', count_conv_layer, 'warstw konwolucyjnych w sieci')
                model = load_model(path_to_model)
                cutted_model = NNModifier.cut_model_to(model, cut_after_layer=i+2)  # i + 2 ponieważ trzeba uwzględnić
                                                                                # jeszcze warstwę normalizującą i ReLU

                for layer in cutted_model.layers:  # Zamrożenie wszystkich warstw w sieci
                    layer.trainable = False

                cutted_model = NNModifier.add_classifier_to_end(cutted_model)

                # wczytanie wag do sieci
                cutted_model.load_weights(path_to_model, by_name=True)
                cutted_model.summary()

                del model  # usunięcie orginalnego modelu z pamięci karty(nie jestem pewny czy go usuwa)

                scores = train_and_asses_network(cutted_model, BATCH_SIZE, count_conv_layer)

                scores.append(count_conv_layer)
                add_score_to_file(scores)

                # tf.reset_default_graph()
                keras.backend.clear_session()

    print('\nSzacowanie skuteczności poszczegulnych warstw sieci zakończone\n')

def train_and_asses_network(cutted_model, BATCH_SIZE, model_ID):
    """Funkcja trenująca i oceniająca skuteczność sieci"""
    optimizer = SGD(lr=0.1, momentum=0.9, nesterov=True)
    # optimizer = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    cutted_model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Wczytanie bazy zdjęć
    [x_train, x_validation, x_test], [y_train, y_validation, y_test] = NNLoader.load_CIFAR10()

    TRAIN_SIZE = len(x_train)
    VALIDATION_SIZE = len(x_validation)

    # Ustawienie ścieżki zapisu i stworzenie folderu jeżeli nie istnieje
    dir_name = str(datetime.datetime.now().strftime("%y-%m-%d %H-%M") +
                   'warstw_' + str(model_ID) + '_konwolucyjnych')
    relative_path_to_save_model = os.path.join('Zapis modelu-uciete/',
                                            datetime.datetime.now().strftime("%y-%m-%d %H-%M"), dir_name)
    absolute_path_to_save_model = os.path.join(os.getcwd(), relative_path_to_save_model)
    if not os.path.exists(absolute_path_to_save_model):  # stworzenie folderu jeżeli nie istnieje
        os.makedirs(absolute_path_to_save_model)

    # Ustawienie ścieżki logów i stworzenie folderu jeżeli nie istnieje
    relative_log_path = 'log/' + str(datetime.datetime.now().strftime("%y-%m-%d %H-%M") + 'warstw_' +
                                     str(model_ID) + '_konwolucyjnych' + '/')
    absolute_log_path = os.path.join(os.getcwd(), relative_log_path)
    if not os.path.exists(absolute_log_path):  # stworzenie folderu jeżeli nie istnieje
        os.makedirs(absolute_log_path)

    # Callback
    learning_rate_regulation = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=1,
                                                                 mode='auto', cooldown=5, min_lr=0.0005,
                                                                 min_delta=0.0005)
    csv_logger = keras.callbacks.CSVLogger('training.log')  # Tworzenie logów
    tensorBoard = keras.callbacks.TensorBoard(log_dir=relative_log_path)  # Wizualizacja uczenia
    modelCheckPoint = keras.callbacks.ModelCheckpoint(  # Zapis sieci podczas uczenia
        filepath=relative_path_to_save_model + "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5", monitor='val_acc',
        save_best_only=True, period=5, save_weights_only=False)
    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)  # zatrzymanie uczenia sieci jeżeli
    # dokładność się nie zwiększa

    print('Using real-time data augmentation.')
    # Agmentacja denych w czasie rzeczywistym
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=True,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=True,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=4,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=4,
        # shear_range=0.1,  # set range for random shear. Pochylenie zdjęcia w kierunku przeciwnym do wskazówek zegara
        # zoom_range=0.1,  # set range for random zoom
        channel_shift_range=0,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=1. / 255,  # Przeskalowanie wejścia
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    val_datagen = ImageDataGenerator(rescale=1. / 255,
                                     samplewise_center=True,  # set each sample mean to 0
                                     samplewise_std_normalization=True,  # divide each input by its std
                                     )
    val_datagen.fit(x_validation)


    cutted_model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),  # Podawanie danych uczących
        verbose=1,
        steps_per_epoch=TRAIN_SIZE // BATCH_SIZE,  # Ilość batchy zanim upłynie epoka
        epochs=1000,  # ilość epok treningu
        callbacks=[csv_logger, tensorBoard, modelCheckPoint, earlyStopping, learning_rate_regulation],
        validation_steps=VALIDATION_SIZE // BATCH_SIZE,
        workers=10,
        validation_data=val_datagen.flow(x_validation, y_validation, batch_size=BATCH_SIZE),
        # use_multiprocessing=True,
        shuffle=True,
        # initial_epoch=1       # Wskazanie od której epoki rozpocząć uczenie
        # max_queue_size=2
    )
    keras.backend.clear_session()
    cutted_model = NNLoader.load_best_weights_from_dir(absolute_path_to_save_model)

    test_generator = ImageDataGenerator(rescale=1. / 255,
                                        samplewise_center=True,  # set each sample mean to 0
                                        samplewise_std_normalization=True,  # divide each input by its std
                                        )

    test_generator.fit(x_validation)
    # keras.backend.get_session().run(tf.global_variables_initializer())
    scores = cutted_model.evaluate_generator(
        test_generator.flow(x_validation, y_validation, batch_size=BATCH_SIZE),
        steps=VALIDATION_SIZE // BATCH_SIZE,
        verbose=1,
    )

    print('Validation loss:', scores[0])
    print('Validation accuracy:', scores[1])
    return scores


def shallow_network(path_to_original_model, path_to_assessing_data='Skutecznosc warstw.json'):
    """Metoda wypłycająca sieć, na podstawie pliku tekstowego  ze ścierzki path_to_assessing_data"""

    print('Wypłycanie sieci')

    # wczytanie danych
    file = open(path_to_assessing_data, "r")
    json_string = file.read()
    layers_accuracy_dict = json.loads(json_string)

    margins = 0.015  # 1.5% dokładności
    last_effective_layer = 1
    layers_to_remove = []
    for i in range(2, len(layers_accuracy_dict)+1):           # Znalezienie warstw do usunięcia
        present_layer = str(i)
        accuracy_diference = layers_accuracy_dict[present_layer]['accuracy'] -\
                             layers_accuracy_dict[str(last_effective_layer)]['accuracy']

        if accuracy_diference < margins:
            layers_to_remove.append(i)
        else:
            last_effective_layer = i + 1

    print('The following convolutional layers and their dependencies(ReLU, batch normalization)will be removed:',
          layers_to_remove, '\n')

    # shallowed_model = load_model('Zapis modelu/19-03-11 20-21/weights-improvement-265-22.78.hdf5',
    #                              custom_objects={'loss_for_knowledge_distillation': loss_for_knowledge_distillation})

    # wczytanie sieci
    original_model = load_model(path_to_original_model)
    shallowed_model = NNModifier.remove_chosen_conv_layers(original_model, layers_to_remove)
    return shallowed_model


def knowledge_distillation(shallowed_model):
    """Metoda Dokonująca transferu danych"""

    print('Knowledge distillation')
    # shallowed_model = NNModifier.add_loss_layer_for_knowledge_distillation(shallowed_model, num_classes=10)
    optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
    shallowed_model.compile(optimizer=optimizer, loss=loss_for_knowledge_distillation)
    # original_model.compile(SGD, loss='categorical_crossentropy', metrics=['accuracy'])


    # Ustawienie ścieżki zapisu i stworzenie folderu jeżeli nie istnieje
    scierzka_zapisu = 'Zapis modelu/' + str(datetime.datetime.now().strftime("%y-%m-%d %H-%M") + '/')
    scierzka_zapisu_dir = os.path.join(os.getcwd(), scierzka_zapisu)
    if not os.path.exists(scierzka_zapisu_dir):  # stworzenie folderu jeżeli nie istnieje
        os.makedirs(scierzka_zapisu_dir)

    # Ustawienie ścieżki logów i stworzenie folderu jeżeli nie istnieje
    scierzka_logow = 'log/' + str(datetime.datetime.now().strftime("%y-%m-%d %H-%M") + '/')
    scierzka_logow_dir = os.path.join(os.getcwd(), scierzka_logow)
    if not os.path.exists(scierzka_logow_dir):  # stworzenie folderu jeżeli nie istnieje
        os.makedirs(scierzka_logow_dir)

    # Callback
    learning_rate_regulation = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=7, verbose=1, mode='auto', cooldown=5, min_lr=0.0005)
    # csv_logger = keras.callbacks.CSVLogger('training.log')                          # Tworzenie logów
    tensorBoard = keras.callbacks.TensorBoard(log_dir=scierzka_logow, write_graph=False)               # Wizualizacja uczenia
    modelCheckPoint = keras.callbacks.ModelCheckpoint(                              # Zapis sieci podczas uczenia
        filepath=scierzka_zapisu + "/weights-improvement-{epoch:02d}-{loss:.2f}.hdf5", monitor='loss',
        save_best_only=True, period=7, save_weights_only=False)
    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)  # zatrzymanie uczenia sieci jeżeli
                                                                                    # dokładność się nie zwiększa


    params = {'dim': (32, 32),
              'batch_size': 128,
              'n_classes': 10,
              'n_channels': 3,
              'shuffle': True,
              'inputs_number': 3}

    # training_gen = DataGenerator(x_data_name='x_train', y_data_name='y_train', data_dir='data/CIFAR10.h5', **params)
    # validation_gen = DataGenerator(x_data_name='x_validation', y_data_name='y_validation', data_dir='data/CIFAR10.h5', **params)

    training_gen = DG_for_kd(x_data_name='x_train', data_dir='data/CIFAR10.h5',
                             dir_to_weights='Zapis modelu/19-03-03 19-24/weights-improvement-238-0.88.hdf5', **params)
    validation_gen = DG_for_kd(x_data_name='x_validation', data_dir='data/CIFAR10.h5',
                               dir_to_weights='Zapis modelu/19-03-03 19-24/weights-improvement-238-0.88.hdf5', **params)

    shallowed_model.fit_generator(generator=training_gen,
                                  use_multiprocessing=True,
                                  workers=10,
                                  epochs=1000,
                                  callbacks=[tensorBoard, modelCheckPoint, earlyStopping, learning_rate_regulation],
                                  initial_epoch=41
                                  )
    # shallowed_model.save('Zapis modelu/shallowed_model.h5')

    shallowed_model = NNModifier.remove_loos_layer(shallowed_model)
    shallowed_model.compile(optimizer=SGD, loss='categorical_crossentropy', metrics=['accuracy'])
    Create_NN_graph.create_NN_graph(shallowed_model, name='temp')

    [x_train, x_validation, x_test], [y_train, y_validation, y_test] = NNLoader.load_CIFAR10()

    scores = shallowed_model.evaluate(x=x_test,
                                      y=y_test,
                                      verbose=1,
                                      )
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    scores = shallowed_model.evaluate(x=x_train,
                                      y=y_train,
                                      verbose=1,
                                      )
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    # b = np.empty(4500)
    # b = shallowed_model.predict_generator(generator=training_gen,
    #                                       workers=4,
    #                                       verbose=1,
    #                                       use_multiprocessing=True)

    # shallowed_model.summary()
    # file = NNLoader.load('temp.txt')
    # print('kot płot')
    #
    # count_nan = 0
    # for i, table in enumerate(b):
    #     for line in table:
    #         if math.isnan(line):
    #             count_nan += 1
    #
    # print('w tabeli jest', count_nan, 'NaN')

if __name__ == '__main__':

    # assesing_conv_layers(path_to_model='Zapis modelu/VGG16-CIFAR10-0.94acc.hdf5', start_from_layer=10)
    shallowed_model = shallow_network('Zapis modelu/VGG16-CIFAR10-0.94acc.hdf5')
    knowledge_distillation(shallowed_model)

