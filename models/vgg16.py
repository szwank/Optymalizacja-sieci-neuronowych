import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, MaxPool2D, Conv2D, Flatten, Dropout, Activation
from keras.models import Model, model_from_json
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras.layers import Softmax
from keras.applications.vgg16 import VGG16
from keras.applications.resnet_v2 import ResNet50V2
import os
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from datetime import datetime
learning_rate = 0.001
batch_size = 16
dataset_location = '/media/dysk/datasets/isic_challenge_2017/'

traindatagen = ImageDataGenerator(
        featurewise_std_normalization=True,
        featurewise_center=True,
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

meangenerator=iter(meandatagen.flow_from_directory(os.path.join(dataset_location, 'valid'),class_mode='binary',classes=['ben','mal'],target_size=(224,224),batch_size=150,shuffle=False))
traindatagen.fit(meangenerator.next()[0])
validdatagen.fit(meangenerator.next()[0])
traingenerator=traindatagen.flow_from_directory(os.path.join(dataset_location, 'train'),class_mode='binary',classes=['ben','mal'],target_size=(224,224),batch_size=batch_size,shuffle=True)
validgenerator=validdatagen.flow_from_directory(os.path.join(dataset_location, 'valid'),class_mode='binary',classes=['ben','mal'],target_size=(224,224),batch_size=batch_size,shuffle=False)
input_tensor = Input(shape=(224,224,3))
model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
x = Flatten()(model.output)
x = Dense(1024, activation='relu')
x=Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x=Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)
model=Model(model.input,x)  


optimizer = SGD(lr = learning_rate)
model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])
now = datetime.now()
tensorboard = TensorBoard(log_dir='../logs/{}'.format(now.strftime("%H:%M:%S")))
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=10, verbose = 1, mode = 'auto')
filepath="epoch_"+"{epoch:02d}"+"_val_loss_"+"{val_loss:.4f}"+"_val_acc_"+"{val_acc:.3f}"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
model.fit_generator(traingenerator, epochs = 500, validation_data = validgenerator, class_weight= [0.61500615, 2.67379679], callbacks = [tensorboard, checkpoint, reduce_lr], workers = 4, use_multiprocessing= True)
keras.backend.clear_session()