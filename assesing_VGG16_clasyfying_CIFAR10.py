from GeneratorStorage.GeneratorDataLoaderFromMemory import GeneratorDataLoaderFromMemory
from NNLoader import NNLoader
from shallowing_NN_v2 import assesing_conv_filters, assesing_conv_layers, shallow_network_based_on_filters, \
    check_integrity_of_score_file, knowledge_distillation
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model, save_model
import keras.backend as K
from keras.optimizers import SGD
from NNModifier import NNModifier
from GeneratorStorage.GeneratorsFlowStorage import GeneratorsFlowStorage
from GeneratorStorage.GeneratorDataLoaderFromDisc import GeneratorDataLoaderFromDisc
from NNHasher import NNHasher
from NNSaver import NNSaver
import os


def get_generators_for_training():
    train_data_generator = ImageDataGenerator(
            samplewise_center=True,  # set each sample mean to 0
            samplewise_std_normalization=True,  # divide each input by its std
            width_shift_range=4,
            height_shift_range=4,
            fill_mode='nearest',
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            rescale=1./255)

    validation_data_generator = ImageDataGenerator(rescale=1. / 255,
                                     samplewise_center=True,  # set each sample mean to 0
                                     samplewise_std_normalization=True,  # divide each input by its std
                                     )

    test_data_generator = validation_data_generator

    training_data = NNLoader.load_CIFAR10()
    data_loader = GeneratorDataLoaderFromMemory(training_data=training_data)

    generators_for_training = GeneratorsFlowStorage(training_generator=train_data_generator,
                                                    validation_generator=validation_data_generator,
                                                    test_generator=test_data_generator,
                                                    generator_data_loader=data_loader)
    return generators_for_training

def get_list_of_files_in_directory(path):
    files = []
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            files.append(file)
    return files


def main():
    train_whole_layers = False
    train_filters = False

    if train_whole_layers is True:
        path = 'Zapis modelu/VGG16-CIFAR10-0.94acc.hdf5'

        generators_for_training = get_generators_for_training()

        assesing_conv_layers(path,
                             generators_for_training=generators_for_training,
                             size_of_clasificator=[10],
                             BATCH_SIZE=64,
                             resume_testing=False,
                             start_from_conv_layer=1)

    if train_filters is True:

        path = 'Zapis modelu/VGG16-CIFAR10-0.94acc.hdf5'

        generators_for_training = get_generators_for_training()

        assesing_conv_filters(path,
                              generators_for_training=generators_for_training,
                              size_of_clasificator=(100, 100, 1),
                              BATCH_SIZE=32,
                              clasificators_trained_at_one_time=32,
                              filters_in_grup_after_division=1,
                              resume_testing=False)



if __name__ == '__main__':
    main()
