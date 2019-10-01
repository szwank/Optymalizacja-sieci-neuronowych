from keras.models import load_model, save_model
from keras.optimizers import SGD
from NNHasher import NNHasher
from NNModifier import NNModifier
from NNSaver import NNSaver
from keras import backend as K
from shallowing_NN_v2 import check_integrity_of_score_file, shallow_network_based_on_filters, knowledge_distillation, \
    shallow_network_based_on_whole_layers_remove_random_filters, shallow_network_based_on_whole_layers
from assesing_cancer_tisue_networks import get_generators_for_training, get_list_of_files_in_directory
from utils.FileMenager import FileManager
from GeneratorStorage.GeneratorDataLoaderFromMemory import GeneratorDataLoaderFromMemory
from GeneratorStorage.GeneratorsFlowStorage import GeneratorsFlowStorage
from NNLoader import NNLoader
from keras.preprocessing.image import ImageDataGenerator
import os
import time

def get_generators_for_training():
    datagen = ImageDataGenerator(
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
        rescale=1. / 255,  # Przeskalowanie wejścia
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0)

    validation_datagen = ImageDataGenerator(
        samplewise_center=True,  # set each sample mean to 0
        samplewise_std_normalization=True,  # divide each input by its std
        rescale=1. / 255,  # Przeskalowanie wejścia
)

    data_loader = GeneratorDataLoaderFromMemory(NNLoader.load_CIFAR10())

    generators_for_training = GeneratorsFlowStorage(training_generator=datagen,
                                                    validation_generator=validation_datagen,
                                                    test_generator=validation_datagen,
                                                    generator_data_loader=data_loader)
    return generators_for_training


def test_model(model, generator):
    optimizer_SGD = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer_SGD, loss='binary_crossentropy', metrics=['accuracy'])
    test_generator = generator.get_test_data_generator_flow(batch_size=128, shuffle=True)
    scores = model.evaluate_generator(test_generator, steps=len(test_generator))
    return scores


def main():

    remove_all_if_bellow = [0.0075, 0.015, 0.03, 0.075]
    increase_by = [0.015, 0.075, 0.15, 0.3]

    my_optymalization = True
    optymalization_with_removing_random_filters = True
    optymalization_from_paper = True

    number_of_repeatrs = 3


    path = 'Zapis modelu'
    network_name = 'VGG16-CIFAR10-0.94acc.hdf5'
    path_to_model = os.path.join(path, network_name)

    model = load_model(path_to_model)
    model_dict = NNModifier.convert_model_to_dictionary(model)
    model_hash = NNHasher.hash_model(model)
    K.clear_session()

    path_to_assesing_file_full_layers = '[[260.33685]]'
    path_to_assesing_file_single_filters = '[[260.33685]]' + 'v2'

    integrity_result = check_integrity_of_score_file(path_to_assesing_file_single_filters, model_dict)

    if integrity_result:
        print("Score file for single filters is correct.")
    else:
        raise (ValueError("File {} have a bug. Result in layers {} aren't correct".format(
            path_to_assesing_file_single_filters, integrity_result)))

    if my_optymalization is True:
        for remove_if_bellow in remove_all_if_bellow:
            for increase_by_ in increase_by:
                for i in range(number_of_repeatrs):
                    shallow_model = shallow_network_based_on_filters(path_to_model,
                                                                     path_to_assesing_file_single_filters,
                                                                     path_to_assesing_file_full_layers,
                                                                     remove_all_filters_if_below=remove_if_bellow,
                                                                     leave_all_filters_if_above=remove_if_bellow + increase_by_
                                                                     )

                    path_to_shallowed_model = 'temp/shallowed_model.hdf5'

                    save_model(shallow_model, path_to_shallowed_model)

                    K.clear_session()

                    generators_for_training = get_generators_for_training()

                    shallowed_model = knowledge_distillation(path_to_shallowed_model, path_to_model,
                                                             generators_for_training,
                                                             temperature=5)

                    scores = test_model(shallowed_model, generators_for_training)

                    directory_name = "".join(
                        ['number_of_parameters_', str(shallowed_model.count_params()),
                         '_remove_if_bellow_', str(remove_if_bellow), '_leave_if_above_',
                         str(remove_if_bellow + increase_by_)])

                    shallowed_model_name = "".join(['Shallowed_model_test_accuracy_', str(scores[1]), '.hdf5'])

                    path_to_shallowed_model = os.path.join(path, 'shallowed_model_removing_filters', directory_name,
                                                           shallowed_model_name)
                    NNSaver.save_model(shallowed_model, path_to_shallowed_model)

                    K.clear_session()

    if optymalization_with_removing_random_filters is True:
        for remove_if_bellow in remove_all_if_bellow:
            for increase_by_ in increase_by:
                for i in range(number_of_repeatrs):
                    shallow_model = shallow_network_based_on_whole_layers_remove_random_filters(path_to_model,
                                                                                                path_to_assesing_file_full_layers,
                                                                                                remove_all_filters_if_below=remove_if_bellow,
                                                                                                leave_all_filters_if_above=remove_if_bellow + increase_by_
                                                                                                )

                    path_to_shallowed_model = 'temp/shallowed_model.hdf5'

                    save_model(shallow_model, path_to_shallowed_model)

                    K.clear_session()

                    generators_for_training = get_generators_for_training()

                    shallowed_model = knowledge_distillation(path_to_shallowed_model, path_to_model,
                                                             generators_for_training,
                                                             temperature=5)

                    scores = test_model(shallowed_model, generators_for_training)

                    directory_name = "".join(
                        ['number_of_parameters_', str(shallowed_model.count_params()),
                         '_remove_if_bellow_', str(remove_if_bellow), '_leave_if_above_',
                         str(remove_if_bellow + increase_by_)])

                    shallowed_model_name = "".join(['Shallowed_model_test_accuracy_', str(scores[1]), '.hdf5'])

                    path_to_shallowed_model = os.path.join(path, 'shallowed_model_removing_random_filters', directory_name,
                                                           shallowed_model_name)
                    NNSaver.save_model(shallowed_model, path_to_shallowed_model)

                    K.clear_session()

    if optymalization_from_paper is True:
        for remove_if_bellow in remove_all_if_bellow:
            for i in range(number_of_repeatrs):
                shallow_model = shallow_network_based_on_whole_layers(path_to_model,
                                                                      path_to_assesing_file_full_layers,
                                                                      remove_layer_if_below=remove_if_bellow)

                path_to_shallowed_model = 'temp/shallowed_model.hdf5'

                save_model(shallow_model, path_to_shallowed_model)

                K.clear_session()

                generators_for_training = get_generators_for_training()

                shallowed_model = knowledge_distillation(path_to_shallowed_model, path_to_model,
                                                         generators_for_training,
                                                         temperature=5)

                scores = test_model(shallowed_model, generators_for_training)

                directory_name = "".join(
                    ['number_of_parameters_', str(shallowed_model.count_params()),
                     '_remove_layer_if_bellow_', str(remove_if_bellow)])

                shallowed_model_name = "".join(['Shallowed_model_test_accuracy_', str(scores[1]), '.hdf5'])

                path_to_shallowed_model = os.path.join(path, 'shallowed_model_removing_whole_layers',
                                                       directory_name,
                                                       shallowed_model_name)
                NNSaver.save_model(shallowed_model, path_to_shallowed_model)

                K.clear_session()


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('Czas wykonywania: {}'.format(end_time - start_time))