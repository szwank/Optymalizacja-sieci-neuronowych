import tensorflow as tf
import tensorflow.keras as keras
from keras.applications.vgg16 import VGG16
from keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, MaxPool2D, Conv2D, Flatten, Dropout, Activation
from keras.models import Model, model_from_json
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras.layers import Softmax
import numpy as np
from keras.utils import plot_model
import datetime
import os
import json
from CreateNN import CreateNN
from NNLoader import NNLoader
from NNSaver import NNSaver

original_network = VGG16(include_top=False, weights='imagenet', input_shape=(32, 32, 3), classes=10)

for layer in original_network.layers[:10]:
    layer.trainable = False



y = Flatten()(original_network.output)
y = Dropout(0.5)(y)
y = Dense(4096)(y)
# y = Dropout(0.10)(y)
y = Dense(4096)(y)
# y = Dropout(0.10)(y)
y = Dense(10)(y)
y = Softmax(name='predictions')(y)
original_network = Model(original_network.input, y)

original_network.summary()
# Wczytanie wag
original_network.load_weights('Zapis modelu/19-02-08 10-25/weights-improvement-115-0.80.hdf5', by_name=True)

# Parametry modelu/ uczenia
BATCH_SIZE = 512
NUM_CLASSES = 10

# Ustawienia kompilera
optimizer = SGD(lr=0.001, momentum=0.9, nesterov=True, decay=1e-6)
# optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
original_network.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# Wczytanie bazy zdjęć
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


# Wielkości zbiorów danych
TRAIN_SIZE = int(0.9 * len(x_train))
VALIDATION_SIZE = int(len(x_train) - TRAIN_SIZE)
TEST_SIZE = int(len(x_test))

# podział zbioru treningowego na treningowy i walidacyjny
# [x_validation, x_train, a] = np.split(x_train, [VALIDATION_SIZE, TRAIN_SIZE + VALIDATION_SIZE])
# [y_validation, y_train, a] = np.split(y_train, [VALIDATION_SIZE, TRAIN_SIZE + VALIDATION_SIZE])
x_validation = x_train[:VALIDATION_SIZE]
x_train = x_train[VALIDATION_SIZE:]
y_validation = y_train[:VALIDATION_SIZE]
y_train = y_train[VALIDATION_SIZE:]

# Zamiana numeru odpowiedzi na macierz labeli
y_train = np_utils.to_categorical(y_train, NUM_CLASSES)
y_validation = np_utils.to_categorical(y_validation, NUM_CLASSES)
y_test = np_utils.to_categorical(y_test, NUM_CLASSES)

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
learning_rate_regulation = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=7, verbose=1, mode='auto', cooldown=7, min_lr=0.00001, min_delta=0.01)
csv_logger = keras.callbacks.CSVLogger('training.log')                          # Tworzenie logów
tensorBoard = keras.callbacks.TensorBoard(log_dir=scierzka_logow)               # Wizualizacja uczenia
modelCheckPoint = keras.callbacks.ModelCheckpoint(                              # Zapis sieci podczas uczenia
    filepath=scierzka_zapisu + "/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5", monitor='val_acc',
    save_best_only=True, period=5, save_weights_only=False)
earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)  # zatrzymanie uczenia sieci jeżeli
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
        rotation_range=90,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        # shear_range=0.1,  # set range for random shear. Pochylenie zdjęcia w kierunku przeciwnym do wskazówek zegara
        zoom_range=0.1,  # set range for random zoom
        channel_shift_range=0.1,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
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
datagen.fit(x_train)

val_datagen = ImageDataGenerator(rescale=1. / 255,
                                 samplewise_center=True,  # set each sample mean to 0
                                 samplewise_std_normalization=True,  # divide each input by its std
                                 )
val_datagen.fit(x_validation)

keras.backend.get_session().run(tf.global_variables_initializer())
original_network.fit_generator(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),  # Podawanie danych uczących
        verbose=1,
        steps_per_epoch=TRAIN_SIZE // BATCH_SIZE,  # Ilość batchy zanim upłynie epoka
        epochs=10000,                         # ilość epok treningu
        callbacks=[csv_logger, tensorBoard, modelCheckPoint, earlyStopping, learning_rate_regulation],
        validation_steps=VALIDATION_SIZE // BATCH_SIZE,
        workers=4,
        validation_data=val_datagen.flow(x_validation, y_validation, batch_size=BATCH_SIZE),
        # use_multiprocessing=True,
        shuffle=True,
        # initial_epoch=10       # Wskazanie od której epoki rozpocząć uczenie
        )

test_generator = ImageDataGenerator(rescale=1. / 255,
                                    samplewise_center=True,  # set each sample mean to 0
                                    samplewise_std_normalization=True,  # divide each input by its std
                                    )

test_generator.fit(x_test)
keras.backend.get_session().run(tf.global_variables_initializer())
scores = original_network.evaluate_generator(
        test_generator.flow(x_test, y_test, batch_size=BATCH_SIZE),
        steps=TEST_SIZE // BATCH_SIZE,
        verbose=1,
        )

print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

