from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Lambda, concatenate
from keras.models import Model, load_model, save_model
from keras.optimizers import SGD
from keras.layers import Softmax
import keras.backend as K
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint, EarlyStopping
import numpy as np
import datetime
import os
from NNModifier import NNModifier
from NNLoader import NNLoader
from Create_NN_graph import Create_NN_graph
from NNHasher import NNHasher
from DataGenerator_for_knowledge_distillation import DataGenerator_for_knowledge_distillation
import json
from custom_loss_function import knowledge_distillation_loos
from custom_metrics import mean_accuracy
from utils.File_menager import FileManager
from custom_metrics import accuracy, soft_categorical_crossentrophy, categorical_crossentropy_metric
from custom_loss_function import loss_for_many_clasificators
from Data_Generator_for_Shallowing import Data_Generator_for_Shallowing
import math


def add_partial_score_to_file(score, file_name, number_of_trained_clasificator):
    """Dopisanie wyniku klasyfikatora do pliku tekstowego."""

    conv_layer_number = number_of_trained_clasificator
    middle_position = int((len(score) - 1) / 2) + 1
    loss = score[1:middle_position]
    accuracy = score[middle_position:]

    if os.path.exists(file_name):
        file = open(file_name, "r")
        json_string = file.read()
        dictionary = json.loads(json_string)
        if str(conv_layer_number) in dictionary.keys():
            dictionary[str(conv_layer_number)]['loss'].extend(loss)
            dictionary[str(conv_layer_number)]['accuracy'].extend(accuracy)
        else:
            subordinate_dictionary = {str(conv_layer_number): {'loss': loss, 'accuracy': accuracy}}
            dictionary.update(subordinate_dictionary)
        file.close()
    else:
        dictionary = {str(conv_layer_number): {'loss': loss, 'accuracy': accuracy}}

    file = open(file_name, "w")
    json_string = json.dumps(dictionary)
    file.write(json_string)
    file.close()

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


def check_if_assesing_chosen_layer_was_complited(file_name: str, model: dict, with_conv_layer: int):
    file = open(file_name, 'r')
    json_string = file.read()
    file.close()

    dictionary = json.loads(json_string)

    layer_number = return_layer_number_of_chosen_conv_layer(model, with_conv_layer)
    number_of_filters_in_last_checked_conv_layer = model["config"]["layers"][layer_number]['config']['filters']

    if len(dictionary[str(with_conv_layer)]['accuracy']) == number_of_filters_in_last_checked_conv_layer:
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

def number_of_filters_in_conv_layer(model: dict, with_conv_layer: int):
    layer_number = return_layer_number_of_chosen_conv_layer(model, with_conv_layer)
    return model["config"]["layers"][layer_number]['config']['filters']

def assesing_conv_layers(path_to_model, start_from_conv_layer=1, BATCH_SIZE=256, clasificators_trained_at_one_time=16,
                         filters_in_grup_after_division=2, resume_testing=False):
    """Metoda oceniająca skuteczność poszczegulnych warstw konwolucyjnych"""

    print('Testowanie warstw konwolucyjnych')
    model = load_model(path_to_model)
    model_hash = NNHasher.hash_model(model)
    score_file_name = model_hash + 'v2'

    model.summary()

    model_architecture = model.to_json(indent=4)
    model_architecture = json.loads(model_architecture)

    if resume_testing is True:
        start_from_conv_layer = check_on_with_layer_testing_was_stopped(score_file_name, model_architecture)
        if not check_if_assesing_chosen_layer_was_complited(score_file_name, model_architecture, start_from_conv_layer):
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

                        cutted_model = NNModifier.add_clssifiers_to_the_all_ends(cutted_model, number_of_classes=10)
                        cutted_model.load_weights(path_to_model, by_name=True)
                        cutted_model.summary()

                        scores = train_and_asses_network(cutted_model, BATCH_SIZE, count_conv_layer)

                        add_partial_score_to_file(score=scores, file_name=score_file_name, number_of_trained_clasificator=count_conv_layer)
                        K.clear_session()

                else:
                    cutted_model = NNModifier.add_clssifiers_to_the_all_ends(cutted_model, number_of_classes=10)
                    cutted_model.load_weights(path_to_model, by_name=True)
                    cutted_model.summary()

                    scores = train_and_asses_network(cutted_model, BATCH_SIZE, count_conv_layer)

                    add_partial_score_to_file(score=scores, file_name=score_file_name,
                                              number_of_trained_clasificator=count_conv_layer)
                K.clear_session()

    print('\nSzacowanie skuteczności poszczegulnych warstw sieci zakończone\n')

def train_and_asses_network(cutted_model, BATCH_SIZE, model_ID):
    number_of_model_outputs = len(cutted_model.outputs)
    optimizer = SGD(lr=0.1, momentum=0.9, nesterov=True)
    # optimizer = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    loss = ['categorical_crossentropy'] * number_of_model_outputs
    loss_weights = [1.0/number_of_model_outputs] * number_of_model_outputs
    # loss_weights[number_of_trained_clasificator] = 1.0
    cutted_model.compile(optimizer,
                         loss=loss,
                         metrics=['accuracy'],
                         loss_weights=loss_weights)
    # Wczytanie bazy zdjęć
    [x_train, x_validation, x_test], [y_train, y_validation, y_test] = NNLoader.load_CIFAR10()

    TRAIN_SIZE = len(x_train)
    VALIDATION_SIZE = len(x_validation)

    # Ustawienie ścieżki zapisu i stworzenie folderu jeżeli nie istnieje
    dir_name = str(datetime.datetime.now().strftime("%y-%m-%d %H-%M") +
                   'warstw_' + str(model_ID) + '_konwolucyjnych')
    relative_path_to_save_model = os.path.join('Zapis modelu-uciete/', dir_name)
    absolute_path_to_save_model = os.path.join(os.getcwd(), relative_path_to_save_model)
    if not os.path.exists(absolute_path_to_save_model):  # stworzenie folderu jeżeli nie istnieje
        os.makedirs(absolute_path_to_save_model)

    # Ustawienie ścieżki logów i stworzenie folderu jeżeli nie istnieje
    relative_log_path = 'log/' + str(datetime.datetime.now().strftime("%y-%m-%d %H-%M") + 'warstw_' +
                                     str(model_ID) + '_konwolucyjnych' + '/')
    absolute_log_path = os.path.join(os.getcwd(), relative_log_path)
    if not os.path.exists(absolute_log_path):  # stworzenie folderu jeżeli nie istnieje
        os.makedirs(absolute_log_path)

    # Callback
    learning_rate_regulation = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1,
                                                                 mode='auto', cooldown=3, min_lr=0.0005,
                                                                 min_delta=0.001)
    tensorBoard = TensorBoard(log_dir=relative_log_path)  # Wizualizacja uczenia
    modelCheckPoint = ModelCheckpoint(  # Zapis sieci podczas uczenia
        filepath=relative_path_to_save_model + "/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss',
        save_best_only=True, period=5, save_weights_only=False)
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  # zatrzymanie uczenia sieci jeżeli
    # dokładność się nie zwiększa

    print('Using real-time data augmentation.')
    # Agmentacja denych w czasie rzeczywistym
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

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).

    val_datagen = ImageDataGenerator(rescale=1. / 255,
                                     samplewise_center=True,  # set each sample mean to 0
                                     samplewise_std_normalization=True,  # divide each input by its std
                                     )
    training_generator = Data_Generator_for_Shallowing(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE), number_of_model_outputs)
    validation_generator = Data_Generator_for_Shallowing(val_datagen.flow(x_validation, y_validation, batch_size=BATCH_SIZE), number_of_model_outputs)


    cutted_model.fit_generator(
        training_generator,  # Podawanie danych uczących
        verbose=1,
        steps_per_epoch=TRAIN_SIZE // BATCH_SIZE,  # Ilość batchy zanim upłynie epoka
        epochs=1000,  # ilość epok treningu
        callbacks=[tensorBoard, modelCheckPoint, earlyStopping, learning_rate_regulation],
        validation_steps=VALIDATION_SIZE // BATCH_SIZE,
        workers=4,
        validation_data=validation_generator,
        use_multiprocessing=True,
        shuffle=True,
        # initial_epoch=1       # Wskazanie od której epoki rozpocząć uczenie
        # max_queue_size=2
    )

    K.clear_session()
    cutted_model = NNLoader.load_best_model_from_dir(absolute_path_to_save_model, mode='lowest')

    test_generator = ImageDataGenerator(rescale=1. / 255,
                                        samplewise_center=True,  # set each sample mean to 0
                                        samplewise_std_normalization=True,  # divide each input by its std
                                        )

    test_generator = Data_Generator_for_Shallowing(val_datagen.flow(x_test, y_test, batch_size=BATCH_SIZE), number_of_model_outputs)
    # keras.backend.get_session().run(tf.global_variables_initializer())
    scores = cutted_model.evaluate_generator(
        test_generator,
        steps=VALIDATION_SIZE // BATCH_SIZE,
        verbose=1,
    )

    K.clear_session()
    print(scores)
    return scores

def shallow_network(path_to_original_model: str, path_to_assessing_data_group_of_filters: str, path_to_assessing_data_full_layers: str="1"):
    """Metoda wypłycająca sieć, na podstawie pliku tekstowego  ze ścierzki path_to_assessing_data"""

    print('Wypłycanie sieci')

    # wczytanie danych
    file = open(path_to_assessing_data_group_of_filters, "r")
    json_string = file.read()
    layers_accuracy_dict = json.loads(json_string)
    file.close()

    accuracy_mean_of_group_of_filters = []
    loss_mean_of_group_of_filters = []

    for i in range(1, len(layers_accuracy_dict)):
        accuracy = layers_accuracy_dict[str(i)]['accuracy']
        loss = layers_accuracy_dict[str(i)]['loss']
        accuracy_mean_of_group_of_filters.append(np.mean(accuracy[:-2]))
        loss_mean_of_group_of_filters.append(np.mean(loss[1:]))

    print(accuracy_mean_of_group_of_filters)
    print(loss_mean_of_group_of_filters)



    margins = 0.015  # 1.5% dokładności
    last_effective_layer = 1
    filters_in_layers_to_remove = {}
    for conv_layer_number in range(1, len(layers_accuracy_dict)):
        filters_to_remove = []
        for number_of_grup_of_filters in range(len(layers_accuracy_dict[str(conv_layer_number)]['accuracy'])):
            if accuracy_mean_of_group_of_filters[conv_layer_number-1] >\
                    layers_accuracy_dict[str(conv_layer_number)]['accuracy'][number_of_grup_of_filters] + margins:
                filters_to_remove.append(number_of_grup_of_filters)

        filters_in_layers_to_remove.update({conv_layer_number: filters_to_remove})

    print(filters_in_layers_to_remove)

    original_model = load_model(path_to_original_model)
    shallowed_model = NNModifier.remove_chosen_filters_from_model(original_model, filters_in_layers_to_remove, 2)
    return shallowed_model

def knowledge_distillation(path_to_shallowed_model, dir_to_original_model):
    """Metoda Dokonująca transferu danych"""

    print('Knowledge distillation')

    # Ustawienie ścieżki zapisu i stworzenie folderu jeżeli nie istnieje
    scierzka_zapisu = 'Zapis modelu/' + str(datetime.datetime.now().strftime("%y-%m-%d %H-%M") + '/')
    FileManager.create_folder(scierzka_zapisu)

    # Ustawienie ścieżki logów i stworzenie folderu jeżeli nie istnieje
    scierzka_logow = 'log/' + str(datetime.datetime.now().strftime("%y-%m-%d %H-%M") + '/')
    FileManager.create_folder(scierzka_logow)

    # Callback
    learning_rate_regulation = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=1, mode='auto', cooldown=5, min_lr=0.0005, min_delta=0.002)
    tensorBoard = TensorBoard(log_dir=scierzka_logow, write_graph=False)               # Wizualizacja uczenia
    modelCheckPoint = ModelCheckpoint(                              # Zapis sieci podczas uczenia
        filepath=scierzka_zapisu + "/weights-improvement-{epoch:02d}-{loss:.2f}.hdf5", monitor='loss',
        save_best_only=True, period=7, save_weights_only=False)
    earlyStopping = EarlyStopping(monitor='val_loss', patience=25)  # zatrzymanie uczenia sieci jeżeli
                                                                                    # dokładność się nie zwiększa

    temperature = 6

    [x_train, x_validation, x_test], [y_train, y_validation, y_test] = NNLoader.load_CIFAR10()

    generator = ImageDataGenerator(rescale=1. / 255,
                                   samplewise_center=True,  # set each sample mean to 0
                                   samplewise_std_normalization=True  # divide each input by its std
                                   )

    training_gen = DataGenerator_for_knowledge_distillation(generator=generator.flow(x_train, y_train, batch_size=64),
                                                            path_to_weights=dir_to_original_model,
                                                            shuffle=True)
    validation_gen = DataGenerator_for_knowledge_distillation(generator=generator.flow(x_validation, y_validation, batch_size=8),
                                                              path_to_weights=dir_to_original_model,
                                                              shuffle=True)

    # validation_gen = DG_for_kd(x_data_name='x_validation', data_dir='data/CIFAR10.h5',
    #                            dir_to_weights=dir_to_original_model, **params)

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

    # shallowed_model.load_weights(path_to_original_model, by_name=True)
    # shallowed_model.load_weights('Zapis modelu/19-05-08 18-39/weights-improvement-28-1.75.hdf5', by_name=True)
    shallowed_model.summary()

    # shallowed_model.load_weights(dir_to_original_model, by_name=True)
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

    # shallowed_model.save('Zapis modelu/shallowed_model.h5')


    # original_model.compile(SGD, loss='categorical_crossentropy', metrics=['accuracy'])
    # shallowed_model = Model(inputs=shallowed_model.inputs, outputs=shallowed_model.outputs[0])
    shallowed_model.compile(optimizer=optimizer_SGD, loss='categorical_crossentropy', metrics=[accuracy,
                                     categorical_crossentropy_metric])
    Create_NN_graph.create_NN_graph(shallowed_model, name='temp')
    [x_train, x_validation, x_test], [y_train, y_validation, y_test] = NNLoader.load_CIFAR10()
    test_generator = DataGenerator_for_knowledge_distillation(generator=generator.flow(x_test, y_test, batch_size=128),
                                                              path_to_weights=dir_to_original_model,
                                                              shuffle=True)

    scores = shallowed_model.evaluate_generator(test_generator)
    print(scores)
    print('Test loss:', scores[2])
    print('Test accuracy:', scores[1])




if __name__ == '__main__':
    path_to_original_model = 'Zapis modelu/VGG16-CIFAR10-0.94acc.hdf5'

    assesing_conv_layers(path_to_model=path_to_original_model,
                         clasificators_trained_at_one_time=16,
                         start_from_conv_layer=1,
                         resume_testing=True)



    # model = load_model(path_to_original_model)
    # model_hash = NNHasher.hash_model(model)
    # K.clear_session()
    # model_architecture = model.to_json(indent=4)
    # model_architecture = json.loads(model_architecture)
    # check_integrity_of_score_file(str(model_hash) + 'v2', model_architecture)

    # shallowed_model = shallow_network(path_to_original_model=path_to_original_model,
    #                                   path_to_assessing_data_group_of_filters=str(model_hash) + 'v2')

    # path_to_shallowed_model = 'temp/model.hdf5'
    # save_model(shallowed_model, filepath=path_to_shallowed_model)
    # K.clear_session()
    #
    # knowledge_distillation(path_to_shallowed_model=path_to_shallowed_model,
    #                        dir_to_original_model=path_to_original_model)
    #
