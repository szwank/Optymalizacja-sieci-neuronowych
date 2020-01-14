# generate outputs of the network with test set as an input
from keras.models import load_model
import numpy as np
import select_gpu
from keras_preprocessing.image import ImageDataGenerator
import os
batch_size = 32
dataset_location = '/media/dysk/datasets/isic_challenge_2017'
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
validdatagen.fit(meangenerator.next()[0])
validgenerator=validdatagen.flow_from_directory(os.path.join(dataset_location, 'test'),class_mode='binary',classes=['ben','mal'],target_size=(224,224),batch_size=batch_size,shuffle=False)

model = load_model('epoch_25_val_loss_0.3064_val_acc_0.860')
predictions = model.predict_generator(validgenerator)
target = validgenerator.classes.reshape(-1,1)
results = np.hstack((target,predictions))
np.save('results', results)