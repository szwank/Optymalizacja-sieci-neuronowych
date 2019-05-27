# %%
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50

import keras
from keras.optimizers import SGD, RMSprop
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D, Input, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
# K.set_image_dim_ordering('th')
import datetime
import math
import glob
import json


def get_model(input_dim, l2_regulizer_weight, leaky_relu_weight):
    input_ = Input(input_dim)
    x = Conv2D(64, (3, 3), padding='same', kernel_initializer='glorot_normal',
               kernel_regularizer=l2(l2_regulizer_weight))(input_)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(leaky_relu_weight)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same', kernel_initializer='glorot_normal',
               kernel_regularizer=l2(l2_regulizer_weight))(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(leaky_relu_weight)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Conv2D(256, (3, 3), padding='same', kernel_initializer='glorot_normal',
               kernel_regularizer=l2(l2_regulizer_weight))(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(leaky_relu_weight)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(512, (3, 3), padding='same', kernel_initializer='glorot_normal',
               kernel_regularizer=l2(l2_regulizer_weight))(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(leaky_relu_weight)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(512, (3, 3), padding='same', kernel_initializer='glorot_normal',
               kernel_regularizer=l2(l2_regulizer_weight))(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(leaky_relu_weight)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(1024, kernel_initializer='glorot_normal', kernel_regularizer=l2(l2_regulizer_weight))(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(leaky_relu_weight)(x)

    x = Dense(1024, kernel_initializer='glorot_normal', kernel_regularizer=l2(l2_regulizer_weight))(x)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(leaky_relu_weight)(x)

    x = Dense(1, kernel_initializer='glorot_normal', kernel_regularizer=l2(l2_regulizer_weight))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('sigmoid')(x)
    return Model(inputs=input_, outputs=x)

def get_callbacks(filepath, save_log_relative_path):
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=10,
                                               verbose=1,
                                               mode='auto',
                                               restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss',
                                  factor=0.1,
                                  patience=3,
                                  verbose=1,
                                  mode='auto',
                                  min_delta=0.0001,
                                  cooldown=2)
    tensorBoard = keras.callbacks.TensorBoard(log_dir=save_log_relative_path)
    return [checkpoint, early_stop, reduce_lr, tensorBoard]

def add_score_to_file(score, zbior, path_to_file):
    """Dopisanie wyniku klasyfikatora do pliku tekstowego."""

    loss = score[0]
    accuracy = score[1]

    if os.path.exists(path_to_file):
        file = open(path_to_file, "r")
        json_string = file.read()
        dictionary = json.loads(json_string)
        subordinate_dictionary = {str(zbior): {'loss': loss, 'accuracy': accuracy}}
        dictionary.update(subordinate_dictionary)
        file.close()
    else:
        dictionary = {str(zbior): {'loss': loss, 'accuracy': accuracy}}

    file = open(path_to_file, "w")
    json_string = json.dumps(dictionary)
    file.write(json_string)
    file.close()

def step_decay(epoch):
    initial_lrate = float(0.001)
    drop = float(0.3)
    epochs_drop = float(10.0)
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    if (epoch % 5) == 0:
        print('Leraning rate: ' + str(lrate))
    lrate32 = np.float32(lrate)
    return float(lrate32)


def main():
    size = 224
    BATCH_SIZE = 32
    reg = 0.0000005
    leak = 0.2

    train_data_generator = ImageDataGenerator(
        featurewise_std_normalization=True,
        featurewise_center=True,
        rotation_range=360,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=[0.9, 1.1],
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        rescale=1.0 / 255)

    validation_data_generator = ImageDataGenerator(
        featurewise_std_normalization=True,
        featurewise_center=True,
        fill_mode='nearest',
        rescale=1.0 / 255)

    test_data_generator = ImageDataGenerator(
        featurewise_std_normalization=True,
        featurewise_center=True,
        fill_mode='nearest',
        rescale=1.0 / 255)

    mean_data_generator = ImageDataGenerator(
        fill_mode='nearest',
        rescale=1.0 / 255)

    # %%
    for iteration in range(1, 7):
        for zbior in range(5):

            meangenerator = mean_data_generator.flow_from_directory('data/' + str(zbior) + '/valid',
                                                                    class_mode='binary',
                                                                    classes=['ben', 'mal'],
                                                                    target_size=(size, size),
                                                                    batch_size=200,
                                                                    shuffle=False)
            data = meangenerator[0]
            train_data_generator.fit(data[0])
            validation_data_generator.fit(data[0])
            test_data_generator.fit(data[0])

            train_generator = train_data_generator.flow_from_directory('data/' + str(zbior) + '/train',
                                                                       class_mode='binary',
                                                                       classes=['ben', 'mal'],
                                                                       target_size=(size, size),
                                                                       batch_size=BATCH_SIZE,
                                                                       shuffle=True,)
            validation_generator = validation_data_generator.flow_from_directory('data/' + str(zbior) + '/valid',
                                                                                 class_mode='binary',
                                                                                 classes=['ben', 'mal'],
                                                                                 target_size=(size, size),
                                                                                 batch_size=BATCH_SIZE,
                                                                                 shuffle=False)


            #
            model = get_model(input_dim=(size, size, 3), l2_regulizer_weight=reg, leaky_relu_weight=leak)
            model.summary()

            # model=load_model(adres[zbior])
            # print('Wczytano: '+adres[zbior])

            # Ustawienie ścieżki logów i stworzenie folderu jeżeli nie istnieje
            save_log_relative_path = os.path.join('log', str(datetime.datetime.now().strftime("%y-%m-%d %H-%M") +
                                                             'collection'+str(zbior)))
            save_log_absolute_path = os.path.join(os.getcwd(), save_log_relative_path)
            if not os.path.exists(save_log_absolute_path):  # stworzenie folderu jeżeli nie istnieje
                os.makedirs(save_log_absolute_path)

            filepath = os.path.join('Zapis modelu', 'Network A', 'iteration' + str(iteration), 'collection' + str(zbior),
                                    "weight_epoch_{epoch:02d}_val_loss_{val_loss:.4f}_val_acc_{val_acc:.3f}")
            if not os.path.exists(filepath):  # stworzenie folderu jeżeli nie istnieje
                os.makedirs(filepath)

            training_callbacks = get_callbacks(filepath, save_log_relative_path)

            opt = SGD(lr=0.01,
                      momentum=0.9,
                      nesterov=True)
            model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
            model.fit_generator(train_generator,
                                steps_per_epoch=24563 // BATCH_SIZE,
                                epochs=5000,
                                validation_data=validation_generator,
                                nb_val_samples=200,
                                callbacks=training_callbacks,
                                use_multiprocessing=True,
                                workers=1)

            test_data = test_data_generator.flow_from_directory('data/' + str(zbior) + '/test',
                                                                                 class_mode='binary',
                                                                                 classes=['ben', 'mal'],
                                                                                 target_size=(size, size),
                                                                                 batch_size=BATCH_SIZE,
                                                                                 shuffle=False)

            score = model.evaluate_generator(test_data,
                                             steps=len(test_data),
                                             use_multiprocessing=True)

            add_score_to_file(score=score, zbior=zbior, path_to_file='scores_of_cancerous_tissue_network.txt')

            K.clear_session()



if __name__ == '__main__':
    main()
