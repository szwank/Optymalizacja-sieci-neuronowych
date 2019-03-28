import tensorflow.keras as keras
import tensorflow as tf
from keras.layers import Input, Dense, MaxPool2D, Conv2D, Flatten, Dropout, Activation, Softmax, Lambda,\
    BatchNormalization, ReLU
from keras import regularizers
from keras.models import Model, model_from_json
from keras.utils import plot_model
from CreateNN import CreateNN
from Create_NN_graph import Create_NN_graph
from NNSaver import NNSaver
import json
import os


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
    def cut_model_to(model, cut_after_layer):
        """Metoda zwraca model ucięty do wskazanej warstwy konwolucyjnej."""
        return Model(model.input, model.layers[cut_after_layer].output)


    @staticmethod
    def add_classifier_to_end(model, number_of_neurons=512, number_of_classes=10, weight_decay=0.0001):

        y = model.output
        y = Flatten()(y)
        y = Dense(number_of_neurons, name='Added_classifier_1')(y)
        y = BatchNormalization(name='Added_normalization_layer')(y)
        y = ReLU(name='Added_ReLU')(y)
        y = Dense(number_of_classes, name='Added_classifier_2', kernel_regularizer=regularizers.l2(weight_decay))(y)
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

        # Inicjalizacja zmiennych
        witch_conv = 1  # Licznik warstw konwolucyjnych
        i = 0   # Licznik warstw sieci

        json_model = model.to_json(indent=4)        # Przekonwertowanie modelu na słownik
        json_object = json.loads(json_model)

        for layer in json_object["config"]["layers"]:  # Iteracja po warstwach w modelu
            if layer["class_name"] == 'Conv2D':     # Sprawdzenie czy warstwa jest konwolucyjna
                if witch_conv in layers_numbers_to_remove:  # sprawdzenie czy warstwa jest na liście warstw do usunięcia

                    # Usunięcie warstwy wraz z odpowiadającymi jej warstwami batch normalization oraz ReLU.
                    json_object = NNModifier.remove_chosen_conv_block_from_json_string(json_object, i)

                witch_conv += 1
            i += 1

        json_model = json.dumps(json_object)# przekonwertowanie słownika z modelem sieci nauronowej spowrotem na model
        model = model_from_json(json_model)

        Create_NN_graph.create_NN_graph(model, name='shallowed_model.png')   # Utowrzenie grafu zmodyfikowanej sieci

        if 'weights_save.h5' in os.listdir('temp/'):  # Usunięcie wag sieci
            os.remove('temp/weights_save.h5')

        return model


    @staticmethod
    def remove_last_layer(model):
        # Rozbicie sieci na warstwy
        layers = [l for l in model.layers]

        input = layers[0].output
        x = input

        for i in range(1, len(layers)-1):
            x = layers[i](x)  # Łączenie warstw sieci neuronowej

        return Model(model.input, x)


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
        loss = Lambda(CreateNN.loss_for_knowledge_distillation, name='loss')([ground_truth, logits,
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


        layer_name = json_object["config"]["layers"][layer_number - 1]["name"]  # Poranie nazwy warstwy porzedającej usuwaną
        json_object["config"]["layers"][layer_number + 1]["inbound_nodes"][0][0][0] = layer_name    # Podmiana połączenia w modelu
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
    def remove_loos_layer(model):
        return Model(inputs=model.input[(2)], outputs=model.layers[-2].output)







