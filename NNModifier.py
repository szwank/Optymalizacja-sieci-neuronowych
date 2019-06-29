import tensorflow.keras as keras
import tensorflow as tf
from keras.layers import Input, Dense, MaxPool2D, Conv2D, Flatten, Dropout, Activation, Softmax, Lambda,\
    BatchNormalization, ReLU, concatenate
from keras import regularizers
from keras.models import Model, model_from_json
from keras.utils import plot_model
from CreateNN import CreateNN
from NNLoader import NNLoader
from Create_NN_graph import Create_NN_graph
from NNSaver import NNSaver
import numpy as np
import json
import os
import keras.backend as K


class NNModifier:



    @staticmethod
    def add_clasyficator_to_each_conv_layer(model):
        # Rozbicie sieci na warstey
        layers = [l for l in model.layers]

        # Inicjalizacja zmiennych
        number_new_outs = 0
        input = layers[0].output
        outputs = []

        x = input

        for i in range(1, len(layers)):
                # x = layers[i](input)
            layers[i].trainable = False
            x = layers[i](x)    # Łączenie warstw sieci neuronowej


            if type(layers[i]) is Conv2D:   # Sprawdzenie czy warstwa jest konwolucyjna.
                                            # Jeżeli jest dodanie klasyfikatora na jej wyjście.
                number_new_outs += 1
                t = Flatten()(x)
                t = Dense(1000)(t)
                t = Dense(10)(t)
                t = Softmax()(t)
                outputs.append(t)            # dodanie wyjścia do listy wyjść

        outputs.append(x)    # Dodanie ostatniego wyjścia

        new_model = Model(input, outputs)

        plot_model(new_model, to_file='new_model.png')      # Debug. Stworzenie rysunku ze strukturą sieci

        return new_model, number_new_outs


    @staticmethod
    def cut_model_to(model, cut_after_layer, leave_layers=[]):
        """Metoda zwraca model ucięty do wskazanej warstwy konwolucyjnej."""
        if not leave_layers:        # sprawdzenie czy lista leave_layers jest pusta
            model.save('temp/model.h5')
            model = Model(model.input, model.layers[cut_after_layer].output)
            model.load_weights('temp/model.h5', by_name=True)
            os.remove('temp/model.h5')
            return model
        else:
            json_model = model.to_json(indent=4)  # Przekonwertowanie modelu na słownik
            json_object = json.loads(json_model)
            removed_layers = 0
            layers_to_remove = []

            for i, layer in enumerate(json_object["config"]["layers"]):  # Iteracja po warstwach w modelu
                if i > cut_after_layer:
                    if not layer["class_name"] in leave_layers:  # Sprawdzenie czy warstwa jest konwolucyjna
                        layers_to_remove.append(i)
                            # Usunięcie warstwy wraz z odpowiadającymi jej warstwami batch normalization oraz ReLU.
                            # json_object = NNModifier.remove_chosen_layer_from_json_string(json_object, i-removed_layers)
                            # removed_layers += 1

            for layer in layers_to_remove:
                # Usunięcie warstwy wraz z odpowiadającymi jej warstwami batch normalization oraz ReLU.
                json_object = NNModifier.remove_chosen_layer_from_json_string(json_object, layer-removed_layers)
                removed_layers += 1

            json_model = json.dumps(json_object)  # przekonwertowanie słownika z modelem sieci nauronowej spowrotem na model
            model = model_from_json(json_model)
            return model

    @staticmethod
    def add_classifier_to_end(model, number_of_neurons=512, number_of_classes=10, weight_decay=0.0001):

        y = model.output
        y = Lambda(lambda x: K.stop_gradient(x))(y)
        y = Flatten()(y)
        y = Dense(number_of_classes, name='Added_classifier_2')(y)
        y = BatchNormalization(name='Added_normalization_layer')(y)
        y = Softmax(name='Added_Softmax')(y)
        return Model(model.input, y)


    @staticmethod
    def remove_chosen_conv_layers(model: Model, layers_numbers_to_remove: list):
        """Metoda usuwa wskazane warstwy konwolucyjne wraz z odpowiadającymi im warstwami Batch normalization oraz ReLU
        jeżeli takie istnieją. Usuwanie odbywa się na słowniku przekonwertowanych z JSON. W konsoli wypisywane są nazwy
        usuwanych warstw. Metoda towrzy grafy sieci przed i po usunięciu warst w folderze temp. Nazwy plików to
        odpowiednio original_model.png oraz shallowed_model.png.

        Argumęty:
        model- model sieci neuronowej która jest usuwana
        layers_numbers_to_remove- numery warstw konwolucyjnych do usunięcia. Warstwy konwolucyjne liczone są kolejno po
        sobie, tzn nie uwzględnia się przy tym innych warstw.

        Zwraza:
        model sieci neurnowej z usuniętymi warstwami.
        """

        # Create_NN_graph.create_NN_graph(model, name='original_model.png')   # Utworzenie grafu orginalnej sieci

        conv_layer_number = 1  # Licznik warstw konwolucyjnych
        number_of_removed_layers = 0

        json_model = model.to_json(indent=4)        # Przekonwertowanie modelu na słownik
        json_object = json.loads(json_model)
        number_of_layers = len(json_object["config"]["layers"])

        for layer_number in range(number_of_layers):  # Iteracja po warstwach w modelu
            layer_number_to_remove = layer_number - number_of_removed_layers
            if json_object["config"]["layers"][layer_number_to_remove]["class_name"] == 'Conv2D':     # Sprawdzenie czy warstwa jest konwolucyjna
                if conv_layer_number in layers_numbers_to_remove:  # sprawdzenie czy warstwa jest na liście warstw do usunięcia

                    # Usunięcie warstwy wraz z odpowiadającymi jej warstwami batch normalization oraz ReLU.
                    json_object = NNModifier.remove_chosen_conv_block_from_json_string(json_object, layer_number_to_remove)
                    number_of_removed_layers += 1
                conv_layer_number += 1

            if len(json_object["config"]["layers"]) <= layer_number:  # Zapewnienie że petla nigdy nie wejdzie na nie istniejące indeksy
                break

        json_model = json.dumps(json_object)# przekonwertowanie słownika z modelem sieci nauronowej spowrotem na model
        model = model_from_json(json_model)

        return model

    @staticmethod
    def remove_last_layer(model):
        return Model(inputs=model.inputs, outputs=model.layers[-2].output)

    @staticmethod
    def remove_chosen_layer_from_json_string(json_object, layer_number):
        """Metoda usuwa wybraną warstwę ze słownika pythonowego. W konsoli wypisywana jest nazwa usuwanej warstwy.

        Argumęty:
        jason_object- słownik pythonowy reprezętujący struktóre sieci neuronowej.
        layer_number- numer warstwy do usunięcia

        Zwraca:
        słownik pythonowy ze zmotyfikowaną strukturą sieci neuronowej."""

        print('deleting', json_object["config"]["layers"][layer_number]["name"])  # wypisanie nazwy usuwanej warstwy

        layer_name = json_object["config"]["layers"][layer_number - 1][
            "name"]  # Poranie nazwy warstwy porzedającej usuwaną

        if layer_number < len(json_object["config"]["layers"]) - 1:
            json_object["config"]["layers"][layer_number + 1]["inbound_nodes"][0][0][0] = layer_name    # Podmiana połączenia w modelu
        else:
            json_object["config"]["output_layers"][0][0] = layer_name

        del json_object["config"]["layers"][layer_number]   # Usunięcia właściwej warstwy ze słownika

        return json_object


    @staticmethod
    def remove_chosen_conv_block_from_json_string(json_object: dict, layer_number: int):
        """Metoda usuwa wskazaną warstwę konwolucyjną i przynależne do niej warstwy batch normalization oraz ReLU.
        W konsoli wypisywane są nazwy usuwanych warstw.

        Argumęty:
        jason_object- słownik pythonowy reprezętujący struktóre sieci neuronowej.
        layer_number- numer warstwy do usunięcia

        Zwraca:
        słownik pythonowy ze zmotyfikowaną strukturą sieci neuronowej."""

        # Wypisanie nazwy usuwanej warstwy konwolucyjnej
        conv_name = json_object["config"]["layers"][layer_number]["name"]  # Pobranie nazwy usuwanej warstwy
        print('Deleting block associated with', conv_name)
        # Sprawdzanie czy trzy następne warstwy to usuwana warstwa konwolucyjna i jej batch normalization oraz ReLU.
        for i in range(3):
            if (json_object["config"]["layers"][layer_number]["class_name"] == 'BatchNormalization' or
                    json_object["config"]["layers"][layer_number]["class_name"] == 'ReLU' or
                    json_object["config"]["layers"][layer_number]["name"] == conv_name):
                json_object = NNModifier.remove_chosen_layer_from_json_string(json_object, layer_number) # Usunięcie warstwy
            else:
                break

        print('Block deleted\n')

        return json_object

    @staticmethod
    def replace_softmax_layer_with_soft_softmax_layer(model, Temperature=100):
        output = Lambda(CreateNN.soft_softmax_layer)(model.layers[-1])
        return Model(inputs=model.inputs, outputs=output)


    @staticmethod
    def remove_loos_layer(model):
        return Model(inputs=model.inputs[2], outputs=model.layers[-2].output)

    @staticmethod
    def rename_choosen_conv_layers(model, chosen_layer_list):
        json_model = model.to_json(indent=4)  # Przekonwertowanie modelu na słownik
        json_object = json.loads(json_model)

        with_conv = 0
        number_of_layers = len(json_object["config"]["layers"])

        for layer_number in range(number_of_layers):  # Iteracja po warstwach w modelu
            if json_object["config"]["layers"][layer_number]["class_name"] == 'Conv2D':     # Sprawdzenie czy warstwa jest konwolucyjna
                if with_conv in chosen_layer_list:  # sprawdzenie czy warstwa jest na liście warstw do usunięcia

                    json_object = NNModifier.add_phrase_to_layer_name(json_object, layer_number, '_changed')
                with_conv += 1

        json_model = json.dumps(json_object)  # przekonwertowanie słownika z modelem sieci nauronowej spowrotem na model
        model = model_from_json(json_model)
        return model


    @staticmethod
    def add_phrase_to_layer_name(json_object, layer_number, phrase):
        old_layer_name = json_object["config"]["layers"][layer_number]["name"]
        new_name = "".join([old_layer_name, phrase])
        json_object["config"]["layers"][layer_number]["name"] = new_name
        json_object["config"]["layers"][layer_number]['config']["name"] = new_name
        json_object["config"]["layers"][layer_number + 1]['inbound_nodes'][0][0][0] = new_name
        return json_object

    @staticmethod
    def split_last_conv_block_on_groups(model, filters_in_grup_after_division, kernel_dimension=(3, 3), pading='same'):
        model.save('temp/model.h5')

        output_layers = []
        conv_layer_name_first_part = 'splited_conv2d_'
        batchnormalization_name_first_part = 'splited_batch_normalization_'
        relu_name_first_part = 'splited_relu_'
        for i in range(1, len(model.layers)+1):
            if 'conv2d' in model.layers[-i].name:
                conv_weights = model.layers[-i].get_weights()
                conv_weights[0] = np.swapaxes(conv_weights[0], 0, 3)
                conv_weights[0] = np.swapaxes(conv_weights[0], 1, 2)
                number_of_filters = model.layers[-i].filters
                if 'batch_normalization' in model.layers[-i+1].name:
                    batch_normalization_weights = model.layers[-i+1].get_weights()

                for j in range(int(conv_weights[0].shape[0]/filters_in_grup_after_division)):
                    x = Conv2D(filters_in_grup_after_division, kernel_dimension, padding=pading,
                               name=conv_layer_name_first_part + str(j))(model.layers[-i - 1].output)
                    x = BatchNormalization(name=batchnormalization_name_first_part + str(j))(x)
                    x = ReLU(name=relu_name_first_part + str(j))(x)
                    output_layers.append(x)
                break

        if output_layers is []:
            raise TypeError("Model don't have convolutional layer")

        model = Model(inputs=model.inputs, outputs=output_layers)
        model.load_weights('temp/model.h5', by_name=True)
        os.remove('temp/model.h5')

        for layer in model.layers:
            layer_name = layer.name
            if conv_layer_name_first_part in layer_name:
                conv_layer_number = int(layer_name[-1])
                start_index = conv_layer_number * filters_in_grup_after_division
                end_index = (conv_layer_number + 1) * filters_in_grup_after_division
                layer_weights = conv_weights[0][start_index:end_index]
                layer_weights = np.swapaxes(layer_weights, 0, 3)
                layer_weights = np.swapaxes(layer_weights, 1, 2)

                layer_biases = conv_weights[1][start_index:end_index]

                layer.set_weights([layer_weights, layer_biases])
                break

            if batchnormalization_name_first_part in layer_name:
                batch_normalization_layer_number = int(layer_name[-1])
                start_index = batch_normalization_layer_number * filters_in_grup_after_division
                end_index = (batch_normalization_layer_number + 1) * filters_in_grup_after_division
                layer_weights = []
                for narray in batch_normalization_weights:
                    layer_weights.append(narray[start_index:end_index])

                layer.set_weights(layer_weights)
                break

        return model

    @staticmethod
    def split_last_conv_block_on_x_groups(model, split_on_x_new_filters, kernel_dimension=(3, 3), pading='same'):

        output_layers = []

        conv_layer_name_first_part = 'splited_conv2d_'
        batch_normalization_name_first_part = 'splited_batch_normalization_'
        relu_name_first_part = 'splited_relu_'
        for i in range(1, len(model.layers) + 1):
            if 'conv2d' in model.layers[-i].name:
                conv_weights = model.layers[-i].get_weights()
                conv_weights[0] = np.swapaxes(conv_weights[0], 0, 3)
                conv_weights[0] = np.swapaxes(conv_weights[0], 1, 2)
                number_of_filters = model.layers[-i].filters
                if 'batch_normalization' in model.layers[-i + 1].name:
                    batch_normalization_weights = model.layers[-i + 1].get_weights()

                for j in range(split_on_x_new_filters):
                    x = Conv2D(int(number_of_filters / split_on_x_new_filters), kernel_dimension, padding=pading,
                               name=conv_layer_name_first_part + str(j))(model.layers[-i - 1].output)
                    x = BatchNormalization(name=batch_normalization_name_first_part + str(j))(x)
                    x = ReLU(name=relu_name_first_part + str(j))(x)
                    output_layers.append(x)
                break

        if output_layers is []:
            raise TypeError("Model don't have convolutional layer")

        model.save('temp/model.h5')
        model = Model(inputs=model.inputs, outputs=output_layers)
        model.load_weights('temp/model.h5', by_name=True)
        os.remove('temp/model.h5')

        for layer in model.layers:
            layer_name = layer.name
            if conv_layer_name_first_part in layer_name:
                conv_layer_number = int(layer_name[-1])
                start_index = conv_layer_number * int(number_of_filters / split_on_x_new_filters)
                end_index = (conv_layer_number + 1) * int(number_of_filters / split_on_x_new_filters)
                layer_weights = conv_weights[0][start_index:end_index]
                layer_weights = np.swapaxes(layer_weights, 0, 3)
                layer_weights = np.swapaxes(layer_weights, 1, 2)

                layer_biases = conv_weights[1][start_index:end_index]

                layer.set_weights([layer_weights, layer_biases])
                break

            if batch_normalization_name_first_part in layer_name:
                batch_normalization_layer_number = int(layer_name[-1])
                start_index = batch_normalization_layer_number * int(number_of_filters / split_on_x_new_filters)
                end_index = (batch_normalization_layer_number + 1) * int(number_of_filters / split_on_x_new_filters)
                layer_weights = []
                for narray in batch_normalization_weights:
                    layer_weights.append(narray[start_index:end_index])

                layer.set_weights(layer_weights)
                break

        return model

    @staticmethod
    def add_clssifiers_to_the_all_ends(model, number_of_classes):
        model.save('temp/model.h5')
        activation_outputs = []
        for i, output in enumerate(model.output):

            y = output
            y = Lambda(lambda x: K.stop_gradient(x))(y)
            y = Flatten()(y)
            y = Dense(number_of_classes, name='Added_classifier_'+str(i))(y)
            y = BatchNormalization(name='Added_normalization_layer_'+str(i))(y)
            y = Softmax(name='Added_Softmax_'+str(i))(y)
            activation_outputs.append(y)

        # output = concatenate(activation_outputs)
        model.load_weights('temp/model.h5', by_name=True)
        os.remove('temp/model.h5')
        model = Model(model.input, activation_outputs)
        return model

    @staticmethod
    def add_concentrate_output_to_the_end(model):
        model.save('temp/model.h5')
        concentrate_output = concatenate(model.outputs)
        model.outputs.append(concentrate_output)
        model = Model(model.inputs, model.outputs)
        model.load_weights('temp/model.h5', by_name=True)

        return model
    @staticmethod
    def remove_chosen_filters_from_model(model, chosen_filters_to_remove: dict, filters_in_grup, debug=False):
        model_dictionary = json.loads(model.to_json())
        number_of_layers = len(model.layers)

        model_weights = []
        last_shallowed_conv_layer_number = 0
        actual_conv_layer_number = 0
        remove_layer_counter = 0

        for layer_number, layer in enumerate(model.layers):
            if type(layer) is Conv2D:
                actual_conv_layer_number += 1

                if actual_conv_layer_number in chosen_filters_to_remove.keys():
                    # Loading weights from layers
                    if actual_conv_layer_number - 1 > last_shallowed_conv_layer_number or model_weights == []:
                        if debug is True:
                            print('Loading weights from {} layer'.format(layer.name))

                        actual_conv_layer_weights = layer.get_weights()
                        actual_conv_layer_weights[0] = NNModifier.swap_axes_in_convonutional_weights(actual_conv_layer_weights[0])
                    else:
                        second_conv_layer_weights[0] = NNModifier.swap_axes_in_convonutional_weights(
                            second_conv_layer_weights[0])
                        actual_conv_layer_weights = second_conv_layer_weights

                    # Sprawdzanie czy niektóre warstwy istnieją po tej konwolucyjnej
                    batch_normalization_layer_is_after_conv_layer = False

                    for i in range(1, 2):
                        if type(model.layers[layer_number+i]) is BatchNormalization:
                            if debug is True:
                                print('Loading weights from {} layer'.format(model.layers[layer_number+i].name))

                            batch_normalization_layer_is_after_conv_layer = True
                            batch_normalization_weights = model.layers[layer_number + i].get_weights()
                            break

                    next_conv_layer_exist = False
                    for i in range(layer_number+1, number_of_layers):
                        if type(model.layers[i]) is Conv2D:
                            if debug is True:
                                print('Loading weights from {} layer'.format(model.layers[i].name))

                            next_conv_layer_exist = True
                            second_conv_layer_weights = model.layers[i].get_weights()
                            second_conv_layer_weights[0] = NNModifier.swap_axes_in_convonutional_weights(
                                second_conv_layer_weights[0])
                            break

                    # Removing the filters weights from the loaded weights
                    actual_filters_to_remove = chosen_filters_to_remove[actual_conv_layer_number]

                    if NNModifier.check_if_list_have_duplicat_arguments(actual_filters_to_remove):
                        raise ValueError("In 'chosen_filters' list's {}th argument, there are duplicates values.".format(actual_conv_layer_number))

                    if len(actual_filters_to_remove) > model_dictionary['config']['layers'][layer_number - remove_layer_counter]['config']['filters']:
                        raise ValueError("In 'chosen_filters' list's {}th argument, there are to many arguments."
                                         "In {}ts layer there are not so many filters to remove.".format(actual_conv_layer_number, actual_conv_layer_number))



                    for removed_weights_x_times, filter_number in enumerate(actual_filters_to_remove):
                        start_index = filter_number * filters_in_grup - removed_weights_x_times * filters_in_grup
                        end_index = start_index + filters_in_grup

                        actual_conv_layer_weights = NNModifier.remove_chosen_weights(weights=actual_conv_layer_weights,
                                                                                     start_index=start_index,
                                                                                     end_index=end_index)
                        model_dictionary['config']['layers'][layer_number - remove_layer_counter]['config']['filters'] -= filters_in_grup


                        if batch_normalization_layer_is_after_conv_layer:
                            batch_normalization_weights = NNModifier.remove_chosen_weights(batch_normalization_weights,
                                                                                           start_index=start_index,
                                                                                           end_index=end_index)

                        if next_conv_layer_exist:
                            second_conv_layer_weights[:1] = NNModifier.remove_chosen_weights(second_conv_layer_weights[:1],
                                                                                             start_index=start_index,
                                                                                             end_index=end_index,
                                                                                             axis=1)

                    if model_dictionary['config']['layers'][layer_number - remove_layer_counter]['config']['filters'] is 0:     # dla usunięcia warstwy
                        NNModifier.remove_chosen_conv_block_from_json_string(model_dictionary, layer_number - remove_layer_counter)
                        remove_layer_counter += 2
                        if batch_normalization_layer_is_after_conv_layer:
                            remove_layer_counter += 1

                    else:       # kiedy warstwa nie jest usunięta
                        actual_conv_layer_weights[0] = NNModifier.swap_axes_in_convonutional_weights(
                            actual_conv_layer_weights[0])
                        if actual_conv_layer_number - 1 > last_shallowed_conv_layer_number or model_weights == []:
                            model_weights.extend(actual_conv_layer_weights)
                        else:
                            model_weights[-2] = actual_conv_layer_weights[0]
                            model_weights[-1] = actual_conv_layer_weights[1]

                        if batch_normalization_layer_is_after_conv_layer:
                            model_weights.extend(batch_normalization_weights)

                        if next_conv_layer_exist:
                            second_conv_layer_weights[0] = NNModifier.swap_axes_in_convonutional_weights(
                                second_conv_layer_weights[0])
                            model_weights.extend(second_conv_layer_weights)

                last_shallowed_conv_layer_number = actual_conv_layer_number

            if actual_conv_layer_number >= max(chosen_filters_to_remove.keys()):
                break


        model = model_from_json(json.dumps(model_dictionary))
        model = NNLoader.load_weights_from_list(model, model_weights, debug)
        return model

    @staticmethod
    def remove_argument_duplicates(the_list: list):
        return list(dict.fromkeys(the_list))


    @staticmethod
    def check_if_list_have_duplicat_arguments(the_list: list):
        seen_arguments = {}
        for argument in the_list:
            if argument in seen_arguments:
                return True
            else:
                seen_arguments.update({argument: None})
        return False

    @staticmethod
    def remove_chosen_weights(weights, start_index, end_index, axis=0):
        """wagi powinny mieć elementy w następującej kolejności:[ilość filtrów, ilość cech wchodzących, wymiar filtra,
        wymiar filtra]"""
        for i, type_weights in enumerate(weights):
            if len(type_weights.shape) > 1:
                weights[i] = np.delete(type_weights, range(start_index, end_index), axis=axis)
            else:
                weights[i] = np.delete(type_weights, range(start_index, end_index), axis=0)
        return weights


    @staticmethod
    def swap_axes_in_convonutional_weights(weights):
        weights = np.swapaxes(weights, 0, 3)
        weights = np.swapaxes(weights, 1, 2)
        return weights

    @staticmethod
    def freeze_all_layers_weights(model: Model):
        for layer in model.layers:
            layer.trainable = False

    @staticmethod
    def remove_choosen_last_conv_blocks(model, start_index, end_index):
        model.save('temp/model.h5')
        model_dictionary = json.loads(model.to_json(indent=4))

        interwal_of_layer_numbers = np.arange(start_index, end_index)
        NNModifier.remove_chosen_conv_block_from_interwal(model_dictionary, interwal_of_layer_numbers)

        NNModifier.remove_chosen_output(model_dictionary, interwal_of_layer_numbers)

        model = model_from_json(json.dumps(model_dictionary))
        model.load_weights('temp/model.h5', by_name=True)
        return model

    @staticmethod
    def remove_chosen_output(model_dictionary: dict, interwal_of_layer_numbers):
        removed_layers = 0
        number_of_elements = len(model_dictionary['config']['output_layers'])
        for i in range(number_of_elements):
            output_name = model_dictionary['config']['output_layers'][i - removed_layers][0]
            if 'splited' in output_name:
                splited_layer_name = output_name.split("_")
                if int(splited_layer_name[-1]) in interwal_of_layer_numbers:
                    del model_dictionary['config']['output_layers'][i - removed_layers]
                    removed_layers += 1

    @staticmethod
    def remove_chosen_conv_block_from_interwal(model_dictionary: dict, interwal_of_layer_numbers):
        removed_layers = 0
        number_of_layers = len(model_dictionary['config']['layers'])
        for layer_number in range(number_of_layers):
            layer_name = model_dictionary['config']['layers'][layer_number - removed_layers]['name']
            if 'splited' in layer_name:
                splited_layer_name = layer_name.split("_")
                if int(splited_layer_name[-1]) in interwal_of_layer_numbers:
                    del model_dictionary['config']['layers'][layer_number - removed_layers]
                    removed_layers += 1