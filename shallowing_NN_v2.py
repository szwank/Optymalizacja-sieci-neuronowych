from keras.preprocessing.image import ImageDataGenerator
from keras.layers import concatenate
from keras.models import Model, load_model, save_model
from keras.optimizers import SGD
from keras.layers import Softmax
import keras.backend as K
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint, EarlyStopping
import datetime
import os
from NNModifier import NNModifier
from NNLoader import NNLoader
from NNHasher import NNHasher
from DataGenerator_for_knowledge_distillation import DataGenerator_for_knowledge_distillation
import json
from custom_loss_function import knowledge_distillation_loos
from utils.FileMenager import FileManager
from custom_metrics import accuracy, soft_categorical_crossentrophy, categorical_crossentropy_metric
import math
import time
from scipy import interpolate
import numpy as np
from GeneratorStorage.GeneratorsFlowStorage import GeneratorsFlowStorage
from GeneratorStorage.GeneratorDataLoaderFromMemory import GeneratorDataLoaderFromMemory
from Data_Generator_for_Shallowing import Data_Generator_for_Shallowing

def add_partial_score_to_file(score, file_name, number_of_trained_clasificator):
    """Dopisanie wyniku klasyfikatora do pliku tekstowego."""

    conv_layer_number = number_of_trained_clasificator
    middle_position = int((len(score) - 1) / 2) + 1
    loss = score[1:middle_position]
    accuracy = score[middle_position:]

    if os.path.exists(file_name):
        while 'json_string' not in locals():
            try:
                with open(file_name, "r") as file:
                    json_string = file.read()
                    file.close()
                    time.sleep(0.5)
            except:
                pass

        dictionary = json.loads(json_string)

        if str(conv_layer_number) in dictionary.keys():
            dictionary[str(conv_layer_number)]['loss'].extend(loss)
            dictionary[str(conv_layer_number)]['accuracy'].extend(accuracy)
        else:
            subordinate_dictionary = {str(conv_layer_number): {'loss': loss, 'accuracy': accuracy}}
            dictionary.update(subordinate_dictionary)

    else:
        dictionary = {str(conv_layer_number): {'loss': loss, 'accuracy': accuracy}}

    writed_correctly = None

    while writed_correctly is None:
        try:
            with open(file_name, "w") as file:
                json_string = json.dumps(dictionary)
                writed_correctly = file.write(json_string)
                file.close()
                time.sleep(0.5)
        except:
            writed_correctly = None


def check_integrity_of_score_file(file_name: str, model: dict):
    file = open(file_name, 'r')
    json_string = file.read()
    file.close()

    dictionary = json.loads(json_string)

    broken_scores_in_conv_layers = []

    for key in dictionary.keys():
        accuracy_len = len(dictionary[key]['accuracy'])
        loss_len = len(dictionary[key]['loss'])

        layer_number = return_layer_number_of_chosen_conv_layer(model, int(key))

        if accuracy_len is not loss_len:
            broken_scores_in_conv_layers.append(int(key))

        elif model["config"]["layers"][layer_number]['config']['filters'] is not accuracy_len:
            broken_scores_in_conv_layers.append(int(key))

    if broken_scores_in_conv_layers is []:
        return True
    else:
        return broken_scores_in_conv_layers


def add_score_to_file(score, file_name, number_of_trained_clasificator):
    """Dopisanie wyniku klasyfikatora do pliku tekstowego."""

    conv_layer_number = score[-1]
    middle_position = int((len(score) - 2) / 2) + 1
    loss = score[:middle_position]
    accuracy = score[middle_position:]

    if os.path.exists(file_name):
        file = open(file_name, "r")
        json_string = file.read()
        dictionary = json.loads(json_string)
        subordinate_dictionary = {str(conv_layer_number): {'loss': loss[1:], 'accuracy': accuracy[:-2]}}
        dictionary.update(subordinate_dictionary)
        file.close()
    else:
        dictionary = {str(conv_layer_number): {'loss': loss[1:], 'accuracy': accuracy[:-2]}}

    file = open(file_name, "w")
    json_string = json.dumps(dictionary)
    file.write(json_string)
    file.close()

def count_layer_by_name(model, key_world_in_name):
    counted_layers = 0
    for layer in model.layers:
        if key_world_in_name in layer.name:
            counted_layers += 1

    return counted_layers

def return_layer_number_of_chosen_conv_layer(model: dict, with_conv_layer: int):
    conv_layer_counter = 0
    for layer_number, layer in enumerate(model["config"]["layers"]):
        if 'conv' in layer['name']:
            conv_layer_counter += 1
            if with_conv_layer == conv_layer_counter:
                return layer_number


def find_max_key_in_dictionary(dictionary):
    return max({int(k) for k in dictionary.keys()})


def remove_choosen_keys(dictionary, keys):
    for key in keys:
        del dictionary[key]


def check_on_with_layer_testing_was_stopped(file_name: str, model: dict):
    file = open(file_name, 'r')
    json_string = file.read()
    file.close()

    dictionary = json.loads(json_string)
    number_of_last_conv_checked = find_max_key_in_dictionary(dictionary)

    return number_of_last_conv_checked


def check_if_assesing_chosen_layer_was_complited(file_name: str, model: dict, with_conv_layer: int, filters_in_groups_after_division: int):
    file = open(file_name, 'r')
    json_string = file.read()
    file.close()

    dictionary = json.loads(json_string)

    layer_number = return_layer_number_of_chosen_conv_layer(model, with_conv_layer)
    number_of_filters_in_last_checked_conv_layer = model["config"]["layers"][layer_number]['config']['filters']

    if len(dictionary[str(with_conv_layer)]['accuracy']) == number_of_filters_in_last_checked_conv_layer/filters_in_groups_after_division:
        return True
    else:
        return False


def remove_scores_of_last_conv_layer(file_name: str):
    file = open(file_name, 'r')
    json_string = file.read()
    file.close()

    dictionary = json.loads(json_string)
    last_conv_layer = find_max_key_in_dictionary(dictionary)
    del dictionary[str(last_conv_layer)]

    json_str = json.dumps(dictionary)
    file = open(file_name, 'w')
    file.write(json_str)
    file.close()

def number_of_filters_in_conv_layer(model: dict, with_conv_layer: int,):
    layer_number = return_layer_number_of_chosen_conv_layer(model, with_conv_layer)
    return model["config"]["layers"][layer_number]['config']['filters']

def assesing_conv_layers(path_to_model, generators_for_training: GeneratorsFlowStorage, size_of_clasificator,
                         start_from_conv_layer=1, BATCH_SIZE=256, clasificators_trained_at_one_time=16,
                         filters_in_grup_after_division=2, resume_testing=False):
    """Metoda oceniająca skuteczność poszczegulnych warstw konwolucyjnych"""

    print('Testowanie warstw konwolucyjnych')
    model = load_model(path_to_model)
    number_of_classes = model.output_shape[1]
    model_hash = NNHasher.hash_model(model)
    score_file_name = model_hash + 'v2'

    model.summary()

    model_architecture = model.to_json(indent=4)
    model_architecture = json.loads(model_architecture)

    if resume_testing is True:
        start_from_conv_layer = check_on_with_layer_testing_was_stopped(score_file_name, model_architecture)
        if not check_if_assesing_chosen_layer_was_complited(score_file_name, model_architecture, start_from_conv_layer, filters_in_grup_after_division):
            remove_scores_of_last_conv_layer(score_file_name)
        else:
            start_from_conv_layer += 1

    del(model)

    count_conv_layer = 0      # Licznik warstw konwolucyjnych.
    number_of_layers_in_model = len(model_architecture["config"]["layers"])

    for i in range(number_of_layers_in_model):

        if model_architecture["config"]["layers"][i]["class_name"] == 'Conv2D':     # Sprawdzenie czy dana warstwa jest konwolucyjna
            count_conv_layer += 1     # Zwiekszenie licznika

            if start_from_conv_layer <= count_conv_layer:
                print('Testowanie', count_conv_layer, 'warstw konwolucyjnych w sieci')
                model = load_model(path_to_model)
                cutted_model = NNModifier.cut_model_to(model, cut_after_layer=i+2)  # i + 2 ponieważ trzeba uwzględnić
                                                                                # jeszcze warstwę normalizującą i ReLU

                cutted_model = NNModifier.split_last_conv_block_on_groups(cutted_model, filters_in_grup_after_division=filters_in_grup_after_division)
                cutted_model.summary()

                del model  # usunięcie orginalnego modelu z pamięci karty(nie jestem pewny czy go usuwa)

                number_of_filters = number_of_filters_in_conv_layer(model_architecture, count_conv_layer)
                number_of_iteration_per_the_conv_layer = math.ceil(number_of_filters / clasificators_trained_at_one_time)

                if number_of_iteration_per_the_conv_layer > 1:      # potrzebne jeżeli ilość trewowanych klasyfikatorów
                    save_model(cutted_model, 'temp/model.hdf5')     # jest wieksza niż ilośc filtrów w warstwie
                    for j in range(number_of_iteration_per_the_conv_layer):
                        cutted_model = load_model('temp/model.hdf5')
                        print('number of clasificator set', j+1, 'of', number_of_iteration_per_the_conv_layer)

                        if j > 0:
                            start_index = 0
                            end_index = j * clasificators_trained_at_one_time
                            cutted_model = NNModifier.remove_choosen_last_conv_blocks(cutted_model, start_index, end_index)

                        if j+1 < number_of_iteration_per_the_conv_layer:
                            start_index = (j+1) * clasificators_trained_at_one_time
                            end_index = number_of_filters
                            cutted_model = NNModifier.remove_choosen_last_conv_blocks(cutted_model, start_index, end_index)

                        for layer in cutted_model.layers:  # Zamrożenie wszystkich warstw w sieci
                            layer.trainable = False

                        cutted_model = NNModifier.add_clssifiers_to_the_all_ends(cutted_model, size_of_clasifier=size_of_clasificator)
                        cutted_model.load_weights(path_to_model, by_name=True)
                        cutted_model.summary()

                        scores = train_and_asses_network(cutted_model, generators_for_training=generators_for_training, batch_size=BATCH_SIZE, model_ID=count_conv_layer)

                        add_partial_score_to_file(score=scores, file_name=score_file_name, number_of_trained_clasificator=count_conv_layer)
                        K.clear_session()

                else:
                    for layer in cutted_model.layers:
                        layer.trainable = False

                    cutted_model = NNModifier.add_clssifiers_to_the_all_ends(cutted_model, size_of_clasifier=size_of_clasificator)
                    cutted_model.load_weights(path_to_model, by_name=True)
                    cutted_model.summary()

                    scores = train_and_asses_network(cutted_model, generators_for_training=generators_for_training, batch_size=BATCH_SIZE, model_ID=count_conv_layer)

                    add_partial_score_to_file(score=scores, file_name=score_file_name,
                                              number_of_trained_clasificator=count_conv_layer)
                K.clear_session()

    print('\nSzacowanie skuteczności poszczegulnych warstw sieci zakończone\n')

def train_and_asses_network(cutted_model, generators_for_training: GeneratorsFlowStorage, batch_size, model_ID):

    optimizer = SGD(lr=0.1, momentum=0.9, nesterov=True)

    number_of_model_outputs = len(cutted_model.outputs)

    number_of_category = generators_for_training.get_train_data_generator_flow(batch_size=batch_size,
                                                                               shuffle=True)[0]

    number_of_category = number_of_category[1].shape[-1]  # geting shape of data

    if number_of_category is 2:
        loss_function = 'binary_crossentropy'
    else:
        loss_function = 'categorical_crossentropy'

    loss = [loss_function] * number_of_model_outputs
    loss_weights = [1.0/number_of_model_outputs] * number_of_model_outputs

    cutted_model.compile(optimizer,
                         loss=loss,
                         metrics=['accuracy'],
                         loss_weights=loss_weights)


    # Ustawienie ścieżki zapisu i stworzenie folderu jeżeli nie istnieje
    dir_name = str(datetime.datetime.now().strftime("%y-%m-%d %H-%M") +
                   'warstw_' + str(model_ID) + '_konwolucyjnych')
    relative_path_to_save_model = os.path.join('Zapis modelu-uciete/', dir_name)
    absolute_path_to_save_model = os.path.join(os.getcwd(), relative_path_to_save_model)
    FileManager.create_folder(relative_path_to_save_model)

    # Ustawienie ścieżki logów i stworzenie folderu jeżeli nie istnieje
    relative_log_path = 'log/' + str(datetime.datetime.now().strftime("%y-%m-%d %H-%M") + 'warstw_' +
                                     str(model_ID) + '_konwolucyjnych' + '/')
    FileManager.create_folder(relative_log_path)

    # Callback
    learning_rate_regulation = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1,
                                                                 mode='auto', cooldown=3, min_lr=0.0005,
                                                                 min_delta=0.001)
    #tensorBoard = TensorBoard(log_dir=relative_log_path)  # Wizualizacja uczenia
    modelCheckPoint = ModelCheckpoint(  # Zapis sieci podczas uczenia
        filepath=relative_path_to_save_model + "/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss',
        save_best_only=True, period=6, save_weights_only=False)
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    train_generator = generators_for_training.get_train_data_generator_flow(batch_size=batch_size,
                                                              shuffle=True)
    train_generator = Data_Generator_for_Shallowing(train_generator, number_of_model_outputs)

    validation_generator = generators_for_training.get_validation_data_generator_flow(batch_size=batch_size,
                                                                                   shuffle=False)

    validation_generator = Data_Generator_for_Shallowing(validation_generator, number_of_model_outputs)

    cutted_model.fit_generator(train_generator,
                               steps_per_epoch=len(train_generator),
                               verbose=1,
                               epochs=1,  # ilość epok treningu
                               callbacks=[modelCheckPoint, earlyStopping, learning_rate_regulation],
                                workers=4,
                                validation_data=validation_generator,
                                validation_steps=len(validation_generator),
                                use_multiprocessing=False,
                                shuffle=True,
    )

    K.clear_session()

    cutted_model = NNLoader.load_best_model_from_dir(absolute_path_to_save_model, mode='lowest')

    test_generator = generators_for_training.get_test_data_generator_flow(batch_size=batch_size,
                                                             shuffle=False)
    test_generator = Data_Generator_for_Shallowing(test_generator)

    scores = cutted_model.evaluate_generator(test_generator,
                                             steps=len(test_generator),
                                             verbose=1,
                                             )

    K.clear_session()
    print(scores)
    return scores

def shallow_network(path_to_original_model: str, path_to_assessing_data_group_of_filters: str, path_to_assessing_data_full_layers: str):
    """Metoda wypłycająca sieć, na podstawie pliku tekstowego  ze ścierzki path_to_assessing_data"""

    print('Wypłycanie sieci')

    filters_accuracy_dict = FileManager.get_dictionary_from_json_text_file(path_to_assessing_data_group_of_filters)
    layers_accuracy_dict = FileManager.get_dictionary_from_json_text_file(path_to_assessing_data_full_layers)


    accuracy_of_whole_layers = []
    for i in range(len(layers_accuracy_dict)):
        accuracy_of_whole_layers.append(layers_accuracy_dict[str(i+1)]['accuracy'])

    remove_all_filters_if_below = 0.015
    leave_all_filters_if_above = 0.1

    accuracy_of_previous_not_removed_layer = 0
    removed_layer_counter = 0
    filters_in_layers_to_remove = {}
    conv_layers_to_remove = []

    for conv_layer_number in range(len(filters_accuracy_dict)):
        filters_to_remove = []

        number_of_filters_in_layer = len(filters_accuracy_dict[str(conv_layer_number+1)]['accuracy'])

        actual_accuracy = accuracy_of_whole_layers[conv_layer_number] - accuracy_of_previous_not_removed_layer
        print('Accuracy increase in {} layer:{} %'.format(conv_layer_number + 1, actual_accuracy*100))
        if actual_accuracy < remove_all_filters_if_below:
            print('layer {} will be fully removed\n'.format(conv_layer_number + 1))

            conv_layers_to_remove.append(conv_layer_number+1)
            removed_layer_counter += 1

        else:
            filters_accuracy_in_actual_layer = {}

            for number_of_grup_of_filters in range(number_of_filters_in_layer):  # corventing
                item = {number_of_grup_of_filters: filters_accuracy_dict[str(conv_layer_number+1)]['accuracy'][number_of_grup_of_filters]}
                filters_accuracy_in_actual_layer.update(item)

            filters_accuracy_in_actual_layer = sort_filters_by_accuracy(filters_accuracy_in_actual_layer)

            percent_of_filters_to_remove = calculate_percent_of_filters_to_remove(actual_accuracy*100, leave_all_filters_if_above*100, remove_all_filters_if_below*100)
            number_of_filters_to_remove = math.floor(number_of_filters_in_layer * percent_of_filters_to_remove)

            print('{} filters in layer {} will be removed\n'.format(number_of_filters_to_remove, conv_layer_number + 1))

            for i in range(number_of_filters_to_remove):
                filters_to_remove.append(filters_accuracy_in_actual_layer[i][0])
                filters_to_remove.sort()

            filters_in_layers_to_remove.update({(conv_layer_number + 1) - removed_layer_counter: filters_to_remove})

            accuracy_of_previous_not_removed_layer = accuracy_of_whole_layers[conv_layer_number]

    print(filters_in_layers_to_remove)

    original_model = load_model(path_to_original_model)
    shallowed_model = NNModifier.rename_choosen_conv_layers(original_model, [x+1 for x in conv_layers_to_remove])
    shallowed_model = NNModifier.rename_first_dense_layer(shallowed_model)
    shallowed_model = NNModifier.remove_chosen_conv_layers(shallowed_model, conv_layers_to_remove)
    shallowed_model.load_weights(path_to_original_model, by_name=True)
    shallowed_model = NNModifier.remove_chosen_filters_from_model(shallowed_model, filters_in_layers_to_remove, 1)
    return shallowed_model


def sort_filters_by_accuracy(filters_accuracy_in_actual_layer: dict):
    filters_accuracy_in_actual_layer = sorted(filters_accuracy_in_actual_layer.items(), key=lambda t: t[1])
    return filters_accuracy_in_actual_layer


def calculate_percent_of_filters_to_remove(argument, leave_all_filters_if_above, remove_all_filters_if_below):
    if argument > leave_all_filters_if_above:
        return 0
    elif argument < remove_all_filters_if_below:
        return 1

    percentage_increases = [0.0, 0.027272727272727258, 0.20909090909090908, 0.3563636363636364, 0.5454545454545454,
                            0.7272727272727273, 1.0, 1.1818181818181819]
    x = calculate_the_values_in_the_range(remove_all_filters_if_below, leave_all_filters_if_above, percentage_increases)
    y = np.array([1, 0.8, 0.31, 0.15, 0.07, 0.02, 0, 0])
    tck = interpolate.splrep(x, y, s=0.001)
    percent = interpolate.splev(argument, tck, der=0)

    if percent < 0:
        return 0
    elif percent > 1:
        return 1
    else:
        return percent


def calculate_the_values_in_the_range(min: float, max: float, increase_value_percents: list):
    values_in_range = []
    for percentage_increase in increase_value_percents:
        value = min + (max - min) * percentage_increase
        values_in_range.append(value)
    return values_in_range


def knowledge_distillation(path_to_shallowed_model,
                           path_to_original_model,
                           generators_for_training: GeneratorsFlowStorage):
    """Metoda dokonująca transferu danych"""

    print('Knowledge distillation')

    # Ustawienie ścieżki zapisu i stworzenie folderu jeżeli nie istnieje
    scierzka_zapisu = 'Zapis modelu/' + str(datetime.datetime.now().strftime("%y-%m-%d %H-%M") + '/')
    FileManager.create_folder(scierzka_zapisu)

    # Ustawienie ścieżki logów i stworzenie folderu jeżeli nie istnieje
    scierzka_logow = 'log/' + str(datetime.datetime.now().strftime("%y-%m-%d %H-%M") + '/')
    FileManager.create_folder(scierzka_logow)

    # Callback
    learning_rate_regulation = ReduceLROnPlateau(monitor='val_categorical_crossentropy_metric', factor=0.1, patience=7, verbose=1, mode='auto', cooldown=5, min_lr=0.0005, min_delta=0.002)
    tensorBoard = TensorBoard(log_dir=scierzka_logow, write_graph=False)               # Wizualizacja uczenia
    modelCheckPoint = ModelCheckpoint(                              # Zapis sieci podczas uczenia
        filepath=scierzka_zapisu + "/weights-improvement-{epoch:02d}-{loss:.2f}.hdf5", monitor='val_categorical_crossentropy_metric',
        save_best_only=True, period=7, save_weights_only=False)
    earlyStopping = EarlyStopping(monitor='val_categorical_crossentropy_metric', patience=15)  # zatrzymanie uczenia sieci jeżeli
                                                                                    # dokładność się nie zwiększa

    temperature = 6

    training_gen = DataGenerator_for_knowledge_distillation(generator=generators_for_training.get_train_data_generator_flow(batch_size=64, shuffle=True),
                                                            path_to_weights=path_to_original_model,
                                                            shuffle=True)
    validation_gen = DataGenerator_for_knowledge_distillation(generator=generators_for_training.get_validation_data_generator_flow(batch_size=8, shuffle=True),
                                                              path_to_weights=path_to_original_model,
                                                              shuffle=True)

    K.clear_session()

    shallowed_model = load_model(path_to_shallowed_model)
    # shallowed_model = CreateNN.get_shallowed_model()
    shallowed_model.layers.pop()

    logits = shallowed_model.layers[-1].output
    probabilieties = Softmax()(logits)

    # logits_T = Lambda(lambda x: x/temperature)(logits)
    # probabilieties_T = Softmax()(logits_T)

    outputs = concatenate([probabilieties, logits])

    shallowed_model = Model(inputs=shallowed_model.inputs, outputs=outputs)

    shallowed_model.summary()

    optimizer_SGD = SGD(lr=0.1, momentum=0.9, nesterov=True)
    shallowed_model.compile(optimizer=optimizer_SGD,
                            loss=knowledge_distillation_loos(alpha_const=0.95, temperature=temperature),
                            metrics=[accuracy,
                                     categorical_crossentropy_metric,
                                     soft_categorical_crossentrophy(temperature)])

    shallowed_model.fit_generator(generator=training_gen,
                                  validation_data=validation_gen,
                                  use_multiprocessing=False,
                                  workers=4,
                                  epochs=1000,
                                  callbacks=[tensorBoard, modelCheckPoint, learning_rate_regulation, earlyStopping],
                                  initial_epoch=0,
                                  max_queue_size=1
                                  )


    shallowed_model.compile(optimizer=optimizer_SGD, loss='categorical_crossentropy', metrics=[accuracy,
                                     categorical_crossentropy_metric])


    test_generator = DataGenerator_for_knowledge_distillation(generator=generators_for_training.get_test_data_generator_flow(batch_size=128, shuffle=True),
                                                              path_to_weights=path_to_original_model,
                                                              shuffle=True)

    scores = shallowed_model.evaluate_generator(test_generator)

    print(scores)
    print('Test loss:', scores[2])
    print('Test accuracy:', scores[1])


if __name__ == '__main__':
    path_to_original_model = 'Zapis modelu/VGG16-CIFAR10-0.94acc.hdf5'

    test = True
    optimalize_network_structure = False
    
    if test is True:
        training_data = NNLoader.load_CIFAR10()
        training_data.get_training_outputs()

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


        generators_for_training = GeneratorsFlowStorage(datagen, train_and_val_datagen, train_and_val_datagen, GeneratorDataLoaderFromMemory(training_data))


        assesing_conv_layers(path_to_model=path_to_original_model,
                             generators_for_training=generators_for_training,
                             size_of_clasificator=(10),
                             clasificators_trained_at_one_time=32,
                             filters_in_grup_after_division=1,
                             start_from_conv_layer=1,
                             resume_testing=False)



    if optimalize_network_structure is True:
        model = load_model(path_to_original_model)
        model_hash = NNHasher.hash_model(model)
        model_architecture = model.to_json(indent=4)
        model_architecture = json.loads(model_architecture)
        K.clear_session()

        check_integrity_of_score_file(str(model_hash) + 'v2', model_architecture)

        shallowed_model = shallow_network(path_to_original_model=path_to_original_model,
                                          path_to_assessing_data_group_of_filters=str(model_hash) + 'v2',
                                          path_to_assessing_data_full_layers=str(model_hash))

        path_to_shallowed_model = 'temp/model.hdf5'
        save_model(shallowed_model, filepath=path_to_shallowed_model)
        K.clear_session()

        generator = ImageDataGenerator(rescale=1. / 255,
                                       samplewise_center=True,  # set each sample mean to 0
                                       samplewise_std_normalization=True  # divide each input by its std
                                       )

        training_data = NNLoader.load_CIFAR10()


        # arguments = {
        #     'class_mode': 'binary',
        #     'classes': ['ben', 'mal'],
        #     'target_size': (224, 224)
        # }

        generators_for_training = GeneratorsFlowStorage(generator, generator, generator, training_data,
                                                        flow_from_directory=False)


        knowledge_distillation(path_to_shallowed_model=path_to_shallowed_model,
                               path_to_original_model=path_to_original_model,
                               generators_for_training=generators_for_training)

