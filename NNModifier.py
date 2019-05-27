import tensorflow.keras as keras
import tensorflow as tf
from keras.layers import Input, Dense, MaxPool2D, Conv2D, Flatten, Dropout, Activation, Softmax, Lambda,\
    BatchNormalization, ReLU, concatenate
from keras import regularizers
from keras.models import Model, model_from_json
from keras.utils import plot_model
from CreateNN import CreateNN
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
    def remove_chosen_conv_layers(model, layers_numbers_to_remove):
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

        Create_NN_graph.create_NN_graph(model, name='original_model.png')   # Utworzenie grafu orginalnej sieci

        witch_conv = 1  # Licznik warstw konwolucyjnych
        number_of_removed_layers = 0

        json_model = model.to_json(indent=4)        # Przekonwertowanie modelu na słownik
        json_object = json.loads(json_model)
        number_of_layers = len(json_object["config"]["layers"])

        for layer_number in range(number_of_layers):  # Iteracja po warstwach w modelu
            layer_number_to_remove = layer_number - number_of_removed_layers
            if json_object["config"]["layers"][layer_number_to_remove]["class_name"] == 'Conv2D':     # Sprawdzenie czy warstwa jest konwolucyjna
                if witch_conv in layers_numbers_to_remove:  # sprawdzenie czy warstwa jest na liście warstw do usunięcia

                    # Usunięcie warstwy wraz z odpowiadającymi jej warstwami batch normalization oraz ReLU.
                    json_object = NNModifier.remove_chosen_conv_block_from_json_string(json_object, layer_number_to_remove)
                    number_of_removed_layers += 1
                witch_conv += 1
                if len(json_object["config"]["layers"]) <= layer_number: # Zapewnienie że petla nigdy nie wejdzie na nie istniejące indeksy
                    break

        json_model = json.dumps(json_object)# przekonwertowanie słownika z modelem sieci nauronowej spowrotem na model
        model = model_from_json(json_model)

        return model


    @staticmethod
    def remove_last_layer(model):
        return Model(inputs=model.inputs, outputs=model.layers[-2].output)


    @staticmethod
    def add_loss_layer_for_knowledge_distillation(model, num_classes, alpha=0.5, T=1):
        """Dodanie warstwy obliczającej loss sieci neuronowej. Warstwa dodawana jest na koniec sieci neuronowej.
        Dodatkowo dodawane są dwa dodatkowe wejścia ground_truth, logits. Są to odpowiedzi sieci orginalnej oraz jej
        wartości sprzed wejścia na softmax. Kolejnośc wejśc w nowej sieci: ground_truth, logits, wejście do
        zoptymalizowanej sieci.

        # Argumenty:
        model- model sieci do której ma być dodana warstwa
        alfa- parametr wyliczania lossu. Waga odpowiadająca za logitsy
        T- temperatura Logitsów

        Zwraca:
        model sieci.
        """

        ground_truth = Input(shape=(10, ), name='ground_truth')  # Dodatkowej wejście na wyjścia z orginalnej sieci(SoftMax)
        logits = Input(shape=(10, ), name='logits')  # Dodatkowe wejście na wyjścia z orginalnej sieci(warstwa przed SoftMax)

        # Warstwa obliczająca loss dla procesu knowledge distillation
        loss = Lambda(CreateNN.loss_for_knowledge_distillation_layer, name='loss')([ground_truth, logits,
                                                                                    model.layers[-1].output,
                                                                                    model.layers[-2].output])
        # label = Softmax(name='test')(model.layers[-2].output)

        model = Model(inputs=(ground_truth, logits, model.input),  outputs=loss)  # dodanie warstwy do modelu
        Create_NN_graph.create_NN_graph(model, name='model_with_loss_layer')
        return model


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
    def remove_chosen_conv_block_from_json_string(json_object, layer_number):
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

        witch_conv = 0
        number_of_layers = len(json_object["config"]["layers"])

        for layer_number in range(number_of_layers):  # Iteracja po warstwach w modelu
            if json_object["config"]["layers"][layer_number]["class_name"] == 'Conv2D':     # Sprawdzenie czy warstwa jest konwolucyjna
                if witch_conv in chosen_layer_list:  # sprawdzenie czy warstwa jest na liście warstw do usunięcia

                    # Usunięcie warstwy wraz z odpowiadającymi jej warstwami batch normalization oraz ReLU.
                    json_object = NNModifier.add_phrase_to_layer_name(json_object, layer_number, '_changed')
                witch_conv += 1

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
    def split_last_conv_block(model, filters_in_grup_after_division=0, split_on_x_new_filters=0, kernel_dimension=(3, 3), pading='same'):
        if filters_in_grup_after_division == 0 and split_on_x_new_filters == 0:
            raise ValueError('Nie podano w jaki sposób podzielić warstwe konwolusyjną. Podaj filters_in_grup_after_division lub split_on_x_new_filters')
        if filters_in_grup_after_division is not 0 and split_on_x_new_filters is not 0:
            raise ValueError(
                'Podaj tylko jeden sposb podziau. Podaj filters_in_grup_after_division lub split_on_x_new_filters')

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

                if filters_in_grup_after_division is not 0:
                    for j in range(int(conv_weights[0].shape[0]/filters_in_grup_after_division)):
                        x = Conv2D(filters_in_grup_after_division, kernel_dimension, padding=pading,
                                   name=conv_layer_name_first_part + str(j))(model.layers[-i - 1].output)
                        x = BatchNormalization(name=batchnormalization_name_first_part + str(j))(x)
                        x = ReLU(name=relu_name_first_part + str(j))(x)
                        output_layers.append(x)
                    break
                else:
                    for j in range(split_on_x_new_filters):
                        x = Conv2D(int(number_of_filters/split_on_x_new_filters), kernel_dimension, padding=pading,
                                   name=conv_layer_name_first_part + str(j))(model.layers[-i - 1].output)
                        x = BatchNormalization(name=batchnormalization_name_first_part + str(j))(x)
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
                if filters_in_grup_after_division is not 0:
                    start_index = conv_layer_number * filters_in_grup_after_division
                    end_index = (conv_layer_number + 1) * filters_in_grup_after_division
                else:
                    start_index = conv_layer_number * int(number_of_filters / split_on_x_new_filters)
                    end_index = (conv_layer_number + 1) * int(number_of_filters / split_on_x_new_filters)
                layer_weights = conv_weights[0][start_index:end_index]
                layer_weights = np.swapaxes(layer_weights, 0, 3)
                layer_weights = np.swapaxes(layer_weights, 1, 2)

                layer_biases = conv_weights[1][start_index:end_index]

                layer.set_weights([layer_weights, layer_biases])

            if batchnormalization_name_first_part in layer_name:
                batch_normalization_layer_number = int(layer_name[-1])
                if filters_in_grup_after_division is not 0:
                    start_index = batch_normalization_layer_number * filters_in_grup_after_division
                    end_index = (batch_normalization_layer_number + 1) * filters_in_grup_after_division
                else:
                    start_index = batch_normalization_layer_number * int(number_of_filters / split_on_x_new_filters)
                    end_index = (batch_normalization_layer_number + 1) * int(number_of_filters / split_on_x_new_filters)
                layer_weights = []
                for narray in batch_normalization_weights:
                    layer_weights.append(narray[start_index:end_index])

                layer.set_weights(layer_weights)

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









