import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, MaxPool2D, Conv2D, Flatten, Dropout, Activation
from keras.models import Model, model_from_json
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.layers import Softmax
import numpy as np
from keras.utils import plot_model
import datetime
import os
import json
from CreateNN import CreateNN
from NNLoader import NNLoader
from NNSaver import NNSaver

def tran_VGG16_on_CIFAR10(batch_size=128, learning_rate=0.1):

    BATCH_SIZE = batch_size

    original_network = CreateNN.create_VGG16_for_CIFAR10()
    original_network.summary()

    # Ustawienia kompilera
    optimizer = SGD(lr=learning_rate, momentum=0.9, nesterov=True, decay=0)
    original_network.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Wczytanie bazy zdjęć
    train_data = NNLoader.load_CIFAR10()

    # Ustawienie ścieżki zapisu i stworzenie folderu jeżeli nie istnieje
    save_model_relative_path = 'Zapis modelu/' + str(datetime.datetime.now().strftime("%y-%m-%d %H-%M") + '/')
    save_model_absolute_path = os.path.join(os.getcwd(), save_model_relative_path)
    if not os.path.exists(save_model_absolute_path):  # stworzenie folderu jeżeli nie istnieje
        os.makedirs(save_model_absolute_path)
    
    # Ustawienie ścieżki logów i stworzenie folderu jeżeli nie istnieje
    save_log_relative_path = 'log/' + str(datetime.datetime.now().strftime("%y-%m-%d %H-%M") + '/')
    save_log_absolute_path = os.path.join(os.getcwd(), save_log_relative_path)
    if not os.path.exists(save_log_absolute_path):  # stworzenie folderu jeżeli nie istnieje
        os.makedirs(save_log_absolute_path)

    learning_rate_regulation = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, verbose=1, mode='auto', cooldown=7, min_lr=0.0005)
    csv_logger = keras.callbacks.CSVLogger('training.log')                          # Tworzenie logów
    tensorBoard = keras.callbacks.TensorBoard(log_dir=save_log_relative_path)               # Wizualizacja uczenia
    modelCheckPoint = keras.callbacks.ModelCheckpoint(                              # Zapis sieci podczas uczenia
        filepath=save_model_relative_path + "/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5", monitor='val_loss',
        save_best_only=True, period=1, save_weights_only=False)
    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)  # zatrzymanie uczenia sieci jeżeli
                                                                                                               # dokładność się nie zwiększa
    
    print('Using real-time data augmentation.')
    # Agmentacja denych w czasie rzeczywistym
    datagen= ImageDataGenerator(
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
            rescale=1./255,  # Przeskalowanie wejścia
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0)
    
    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    
    
    val_datagen = ImageDataGenerator(rescale=1. / 255,
                                     samplewise_center=True,  # set each sample mean to 0
                                     samplewise_std_normalization=True,  # divide each input by its std
                                     )
    
    # keras.backend.get_session().run(tf.global_variables_initializer())
    original_network.fit_generator(
            datagen.flow(train_data.get_training_inputs(), train_data.get_training_outputs(), batch_size=BATCH_SIZE),  # Podawanie danych uczących
            verbose=1,
            steps_per_epoch=len(train_data.get_training_inputs()) // BATCH_SIZE,  # Ilość batchy zanim upłynie epoka
            epochs=1000,                         # ilość epok treningu
            callbacks=[csv_logger, tensorBoard, modelCheckPoint, earlyStopping, learning_rate_regulation],
            validation_steps=len(train_data.get_validation_outputs()) // BATCH_SIZE,
            workers=10,
            validation_data=val_datagen.flow(train_data.get_validation_inputs(), train_data.get_validation_outputs(), batch_size=BATCH_SIZE),
            use_multiprocessing=False,
            shuffle=True,
            # initial_epoch=1     # Wskazanie od której epoki rozpocząć uczenie
            )
    original_network = NNLoader.load_best_weights_from_dir(original_network, save_model_absolute_path)
    return original_network

def asses_model_on_CIFAR10(model, BATCH_SIZE=128):

    train_data = NNLoader.load_CIFAR10()
    TEST_SIZE = len(train_data.get_test_outputs())

    val_generator = ImageDataGenerator(rescale=1. / 255,
                                        samplewise_center=True,  # set each sample mean to 0
                                        samplewise_std_normalization=True,  # divide each input by its std
                                        )

    scores = model.evaluate_generator(
            val_generator.flow(train_data.get_test_inputs(), train_data.get_training_outputs(), batch_size=BATCH_SIZE),
            steps=TEST_SIZE // BATCH_SIZE,
            verbose=1,
            )
    
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


if __name__ == '__main__':
    model = tran_VGG16_on_CIFAR10(64)
    asses_model_on_CIFAR10(model)

    