from pathlib import PurePath, Path

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, MaxPool2D, Conv2D, Flatten, Dropout, Activation
from keras.models import Model, model_from_json
from keras.optimizers import SGD, Adam
from keras.applications.vgg16 import VGG16
import os
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from datetime import datetime

learning_rate = 0.01
batch_size = 16
dataset_path = PurePath(Path().resolve())
dataset_path = dataset_path.parent.joinpath('data')


train_data_gen = ImageDataGenerator(
    featurewise_std_normalization=True,
    featurewise_center=True,
    zoom_range=[0.9, 1.1],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    rescale=1.0 / 255)

valid_data_gen = ImageDataGenerator(
    featurewise_std_normalization=True,
    featurewise_center=True,
    fill_mode='nearest',
    rescale=1.0 / 255)

mean_data_gen = ImageDataGenerator(rescale=1.0 / 255)

mean_generator = mean_data_gen.flow_from_directory(dataset_path.joinpath('train'),
                                                   class_mode='binary',
                                                   target_size=(224, 224),
                                                   batch_size=500,
                                                   shuffle=True)  # We have to shuffle dataset, otherwise mean and std can be taken from one class
train_data_gen.fit(mean_generator.next()[0])
# this way is more confident and faster
valid_data_gen.mean = train_data_gen.mean
valid_data_gen.std = train_data_gen.std

train_generator = train_data_gen.flow_from_directory(dataset_path.joinpath('train'), class_mode='binary',
                                                     classes=['dog', 'cat'], target_size=(224, 224),
                                                     batch_size=batch_size, shuffle=True)

valid_generator = valid_data_gen.flow_from_directory(dataset_path.joinpath('valid'), class_mode='binary',
                                                     classes=['dog', 'cat'], target_size=(224, 224),
                                                     batch_size=batch_size, shuffle=False)

input_tensor = Input(shape=(224, 224, 3))
model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
x = Flatten()(model.output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.1)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.1)(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(input_tensor, x)

i = 0
while type(model.layers[i]) is not Dense:
    model.layers[i].trainable = False
    i += 1

optimizer = SGD(lr=learning_rate, nesterov=True, momentum=0.9)

model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])
now = datetime.now()
tensorboard = TensorBoard(log_dir='../logs/{}'.format(now.strftime("%H:%M:%S")))
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=10, verbose=1, mode='auto')
filepath = "epoch_" + "{epoch:02d}" + "_val_loss_" + "{val_loss:.4f}" + "_val_acc_" + "{val_acc:.3f}"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
early_stop = EarlyStopping(patience=15,)

model.fit_generator(train_generator, epochs=10,
                    validation_data=valid_generator,
                    class_weight=[0.61500615, 2.67379679],
                    callbacks=[tensorboard, checkpoint, reduce_lr, early_stop],
                    workers=4,
                    use_multiprocessing=True)

for layer in model.layers:
    layer.trainable = True

learning_rate = 0.001
optimizer = SGD(lr=learning_rate, nesterov=True, momentum=0.9)
model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])

model.fit_generator(train_generator, epochs=500,
                    validation_data=valid_generator,
                    class_weight=[0.61500615, 2.67379679],
                    callbacks=[tensorboard, checkpoint, reduce_lr, early_stop],
                    workers=4,
                    use_multiprocessing=True)