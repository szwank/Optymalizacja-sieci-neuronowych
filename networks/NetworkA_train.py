#%% 
import os
import numpy as np

import keras
from keras.models import Sequential, load_model
from keras.optimizers import SGD, RMSprop
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

import datetime
import math


size=224

def step_decay(epoch):
    initial_lrate = float(0.001)
    drop = float(0.3)
    epochs_drop = float(10.0)
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    if (epoch%5)==0:
        print('Leraning rate: '+str(lrate))
    lrate32=np.float32(lrate)
    return float(lrate32)




    
    
traindatagen = ImageDataGenerator(
        featurewise_std_normalization=True,
        featurewise_center=True,
        rotation_range=360,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=[0.9,1.1],
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        rescale=1.0/255)


validdatagen = ImageDataGenerator(
        featurewise_std_normalization=True,
        featurewise_center=True,
        fill_mode='nearest',
        rescale=1.0/255)
meandatagen = ImageDataGenerator(
        fill_mode='nearest',
        rescale=1.0/255)


def create_model(reg=0.0000005, leak=0.2):

    model = Sequential()
    model.add(Convolution2D(64, (3, 3), border_mode='same', init='glorot_normal', W_regularizer=l2(reg),
                            input_shape=(3, size, size)))
    model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU(leak))

    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(128, (3, 3), border_mode='same', init='glorot_normal', W_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU(leak))

    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(256, (3, 3), border_mode='same', init='glorot_normal', W_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU(leak))

    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(512, (3, 3), border_mode='same', init='glorot_normal', W_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU(leak))

    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(512, (3, 3), border_mode='same', init='glorot_normal', W_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU(leak))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, init='glorot_normal', W_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU(leak))
    model.add(Dense(1024, init='glorot_normal', W_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=-1))
    model.add(LeakyReLU(leak))
    model.add(Dense(1, init='glorot_normal', W_regularizer=l2(reg)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('sigmoid'))
    model.summary()

    return model


for ess in range(1,7):
    for zbior in range(5):
    
        meangenerator=meandatagen.flow_from_directory('../data/'+str(zbior)+'/valid',
                                                      class_mode='binary',
                                                      classes=['ben','mal'],
                                                      target_size=(size,size),
                                                      batch_size=200,
                                                      shuffle=False)
        for data in meangenerator:
            break
        data = data[0]
        traindatagen.fit(data)
        validdatagen.fit(data)
               
        traingenerator=traindatagen.flow_from_directory('../data/'+str(zbior)+'/train',
                                                        class_mode='binary',
                                                        classes=['ben', 'mal'],
                                                        target_size=(size, size),
                                                        batch_size=16,
                                                        shuffle=True)

        validgenerator=validdatagen.flow_from_directory('../data/'+str(zbior)+'/valid',
                                                        class_mode='binary',
                                                        classes=['ben', 'mal'],
                                                        target_size=(size, size),
                                                        batch_size=16,
                                                        shuffle=False)

        model = create_model()

        filepath="../models/Network A/ess"+str(ess)+"/fold"+str(zbior)+"_epoch_"+"{epoch:02d}"+"_val_loss_"+"{val_loss:.4f}"+"_val_acc_"+"{val_acc:.3f}"

        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')
        opt = SGD(lr=0.001,momentum=0.9, nesterov=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', epsilon=0.0001, cooldown=0)

        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        model.fit_generator(traingenerator,
                            samples_per_epoch=24563,
                            nb_epoch=5000,
                            validation_data=validgenerator,
                            nb_val_samples=200,
                            callbacks=[checkpoint, early_stop, reduce_lr])
        K.clear_session()

#%%




    
