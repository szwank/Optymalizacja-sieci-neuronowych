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
    optimize_networks = False

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



    if optimize_networks is True:

        path = 'Zapis modelu/VGG16-CIFAR10-0.94acc.hdf5'

        model = load_model(path)
        model_dict = NNModifier.convert_model_to_dictionary(model)
        model_hash = NNHasher.hash_model(model)
        K.clear_session()

        path_to_assesing_file_full_layers = model_hash
        path_to_assesing_file_single_filters = model_hash + 'v2'

        integrity_result = check_integrity_of_score_file(path_to_assesing_file_single_filters, model_dict)

        if integrity_result:
            print("Score file for single filters is correct.")
        else:
            raise (ValueError("File {} have a bug. Result in layers {} aren't correct".format(
                path_to_assesing_file_single_filters, integrity_result)))

        shallow_model = shallow_network_based_on_filters(path, path_to_assesing_file_single_filters,
                                                         path_to_assesing_file_full_layers)

        path_to_shallowed_model = 'temp/shallowed_model.hdf5'

        save_model(shallow_model, path_to_shallowed_model)

        K.clear_session()

        generators_for_training = get_generators_for_training(zbior)

        shallowed_model = knowledge_distillation(path_to_shallowed_model, path, generators_for_training)

        optimizer_SGD = SGD(lr=0.01, momentum=0.9, nesterov=True)
        shallowed_model.compile(optimizer=optimizer_SGD, loss='binary_crossentropy', metrics=['accuracy'])
        test_generator = generators_for_training.get_test_data_generator_flow(batch_size=128, shuffle=True)
        scores = shallowed_model.evaluate_generator(test_generator, steps=len(test_generator))

        shallowed_model_name = "".join(
            ['Shallowed_model_number_of_parameters_', str(shallowed_model.count_params()), '_test_accuracy_', str(scores[1]), '.hdf5'])
        path_to_shallowed_model = os.path.join(path, 'shallowed_model', shallowed_model_name)
        NNSaver.save_model(shallowed_model, path_to_shallowed_model)


if __name__ == '__main__':
    main()
