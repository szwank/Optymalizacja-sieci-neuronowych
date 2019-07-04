from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.optimizers import SGD
from keras.datasets import cifar10
from NNLoader import NNLoader
from GeneratorStorage.GeneratorsFlowStorage import GeneratorsFlowStorage
from keras.utils import np_utils
from GeneratorStorage.GeneratorDataLoaderFromMemory import GeneratorDataLoaderFromMemory
from GeneratorStorage.GeneratorDataLoaderFromDisc import GeneratorDataLoaderFromDisc
import os

path_to_original_model = 'Zapis modelu/VGG16-CIFAR10-0.94acc.hdf5'

test = True
optimalize_network_structure = False

if test is True:
    training_data = NNLoader.load_CIFAR10()

    datagen = ImageDataGenerator(
        samplewise_center=True,  # set each sample mean to 0
        samplewise_std_normalization=True,  # divide each input by its std
        width_shift_range=4,
        height_shift_range=4,
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        rescale=1. / 255,  # Przeskalowanie wejścia
    )

    train_and_val_datagen = ImageDataGenerator(rescale=1. / 255,
                                               samplewise_center=True,  # set each sample mean to 0
                                               samplewise_std_normalization=True,  # divide each input by its std
                                               )
    data_loader = GeneratorDataLoaderFromDisc(os.path.join('data', '1', 'train'), os.path.join('data', '1', 'valid'), os.path.join('data', '1', 'test'),
                                          'binary', ['ben', 'mal'], (224, 224))

    generators_for_training = GeneratorsFlowStorage(datagen, train_and_val_datagen, train_and_val_datagen,
                                                    GeneratorDataLoaderFromMemory(training_data, 32))

    model = load_model(path_to_original_model, compile=False)
    model.compile(SGD(lr=0.1, momentum=0.9, nesterov=True),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    generator = generators_for_training.get_train_data_generator_flow(batch_size=64, shuffle=True)


    model.fit_generator(generator=generators_for_training.get_train_data_generator_flow(batch_size=64, shuffle=True), steps_per_epoch=10,
                        validation_data=generators_for_training.get_validation_data_generator_flow(64, True), validation_steps=10)