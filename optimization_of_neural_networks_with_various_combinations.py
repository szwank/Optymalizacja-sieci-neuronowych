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
import os


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

    number_of_repeatrs = 6

    for zbior in range(0, 1):
        path = os.path.join('NetworkA', 'fold' + str(zbior))
        network_name = get_list_of_files_in_directory(path)[0]
        path_to_model = os.path.join(path, network_name)

        model = load_model(path_to_model)
        model_dict = NNModifier.convert_model_to_dictionary(model)
        model_hash = NNHasher.hash_model(model)
        K.clear_session()

        path_to_assesing_file_full_layers = os.path.join('NetworkA', model_hash)
        path_to_assesing_file_single_filters = os.path.join('NetworkA', model_hash + 'v2')

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
                                                                         leave_all_filters_if_above=remove_if_bellow + increase_by_)

                        path_to_shallowed_model = 'temp/shallowed_model.hdf5'

                        save_model(shallow_model, path_to_shallowed_model)

                        K.clear_session()

                        generators_for_training = get_generators_for_training(zbior)

                        shallowed_model = knowledge_distillation(path_to_shallowed_model, path_to_model,
                                                                 generators_for_training)

                        scores = test_model(shallowed_model, generators_for_training)

                        directory_name = "".join(
                            ['number_of_parameters_', str(shallowed_model.count_params()),
                             '_remove_if_bellow_', remove_if_bellow, '_leave_if_above_',
                             remove_if_bellow + increase_by_])

                        shallowed_model_name = "".join(['Shallowed_model_test_accuracy_', str(scores[1]), '.hdf5'])

                        path_to_shallowed_model = os.path.join(path, 'shallowed_model_removing_filters', directory_name,
                                                               shallowed_model_name)
                        NNSaver.save_model(shallowed_model, path_to_shallowed_model)

        if optymalization_with_removing_random_filters is True:
            for remove_if_bellow in remove_all_if_bellow:
                for increase_by_ in increase_by:
                    for i in range(number_of_repeatrs):
                        shallow_model = shallow_network_based_on_whole_layers_remove_random_filters(path_to_model,
                                                                                                    path_to_assesing_file_full_layers,
                                                                                                    remove_all_filters_if_below=remove_if_bellow,
                                                                                                    leave_all_filters_if_above=remove_if_bellow + increase_by_)

                        path_to_shallowed_model = 'temp/shallowed_model.hdf5'

                        save_model(shallow_model, path_to_shallowed_model)

                        K.clear_session()

                        generators_for_training = get_generators_for_training(zbior)

                        shallowed_model = knowledge_distillation(path_to_shallowed_model, path_to_model,
                                                                 generators_for_training)

                        scores = test_model(shallowed_model, generators_for_training)

                        directory_name = "".join(
                            ['number_of_parameters_', str(shallowed_model.count_params()),
                             '_remove_if_bellow_', remove_if_bellow, '_leave_if_above_',
                             remove_if_bellow + increase_by_])

                        shallowed_model_name = "".join(['Shallowed_model_test_accuracy_', str(scores[1]), '.hdf5'])

                        path_to_shallowed_model = os.path.join(path, 'shallowed_model_removing_random_filters', directory_name,
                                                               shallowed_model_name)
                        NNSaver.save_model(shallowed_model, path_to_shallowed_model)

        if optymalization_from_paper is True:
            for remove_if_bellow in remove_all_if_bellow:
                for i in range(number_of_repeatrs):
                    shallow_model = shallow_network_based_on_whole_layers(path_to_model,
                                                                          path_to_assesing_file_full_layers,
                                                                          remove_layer_if_below=remove_if_bellow)

                    path_to_shallowed_model = 'temp/shallowed_model.hdf5'

                    save_model(shallow_model, path_to_shallowed_model)

                    K.clear_session()

                    generators_for_training = get_generators_for_training(zbior)

                    shallowed_model = knowledge_distillation(path_to_shallowed_model, path_to_model,
                                                             generators_for_training)

                    scores = test_model(shallowed_model, generators_for_training)

                    directory_name = "".join(
                        ['number_of_parameters_', str(shallowed_model.count_params()),
                         '_remove_layer_if_bellow_', remove_if_bellow])

                    shallowed_model_name = "".join(['Shallowed_model_test_accuracy_', str(scores[1]), '.hdf5'])

                    path_to_shallowed_model = os.path.join(path, 'shallowed_model_removing_whole_layers',
                                                           directory_name,
                                                           shallowed_model_name)
                    NNSaver.save_model(shallowed_model, path_to_shallowed_model)


if __name__ == '__main__':
    main()
