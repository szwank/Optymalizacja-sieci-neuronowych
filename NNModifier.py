import tensorflow.keras as keras
import tensorflow as tf
from keras.layers import Input, Dense, MaxPool2D, Conv2D, Flatten, Dropout, Activation, Softmax, Lambda
from keras.models import Model, model_from_json
from keras.utils import plot_model
from CreateNN import CreateNN
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
    def cut_model_to(model, layer):

        return Model(model.input, model.layers[layer].output)

    @staticmethod
    def add_classifier_to_end(model, number_of_neurons=512, number_of_classes=10):

        y = model.output
        y = Flatten()(y)
        y = Dense(number_of_neurons)(y)
        y = Dense(number_of_classes)(y)
        y = Softmax()(y)

        return Model(model.input, y)

    @staticmethod
    def remove_chosen_conv_layers(model, layers_to_remove):

        # Rozbicie sieci na warstey
        # layers = [l for l in model.layers]
        # model.summary()
        # input = layers[0].output
        if not os.path.exists('temp'):  # stworzenie folderu jeżeli nie istnieje
            os.makedirs('temp')
        plot_model(model, to_file='temp/original_model.png', show_shapes=True)
        witch_conv = 1  # Licznik warstw konwolucyjnych
        i = 0

        json_model = model.to_json(indent=4)        # przekonwertowanie modelu na dict
        json_object = json.loads(json_model)
        for layer in json_object["config"]["layers"]:
            if layer["class_name"] == 'Conv2D':
                if witch_conv in layers_to_remove:
                    json_object = NNModifier.remove_chosen_conv_block_from_json_string(json_object, i)


                    # i -= 1      # zmniejszyła się lidzba warstw
                witch_conv += 1
            i += 1

        # json_object["config"]["layers"][1]["inbound_nodes"][0][0][0]
        json_model = json.dumps(json_object)
        model = model_from_json(json_model)
        model.summary()
        plot_model(model, to_file='temp/modified_model.png', show_shapes=True)
        return model





    @staticmethod
    def remove_last_layer(model):
        # Rozbicie sieci na warstey
        layers = [l for l in model.layers]

        input = layers[0].output
        x = input

        for i in range(1, len(layers)-1):
            x = layers[i](x)  # Łączenie warstw sieci neuronowej

        return Model(model.input, x)



    @staticmethod
    def add_loss_layer_for_knowledge_distillation(model, alpha=0.5, T=1):
        """Dodanie warstwy obliczającej loss sieci neuronowej. Warstwa dodawana jest na koniec sieci neuronowej.
        Dodatkowo dodawane są dwa dodatkowe wejścia ground_truth, logits. Są to odpowiedzi sieci orginalnej oraz jej
        wartości sprzed wejścia na softmax.

        # Argumenty:
        model- model sieci do której ma być dodana warstwa
        alfa- parametr wyliczania lossu. Waga odpowiadająca za logitsy
        T- temperatura Logitsów

        Zwraza:
        model sieci"""

        ground_truth = Input(shape=10)
        logits = Input(shape=10)

        loss = Lambda(CreateNN.loss_for_knowledge_distillation(alpha=alpha, T=T), name='loss')([ground_truth, logits,
                                                                                model.layers[-1].outputs,
                                                                                model.layers[-2].outputs])

        model = Model(model.inputs, loss)
        return model


    @staticmethod
    def remove_chosen_layer_from_json_string(json_object, layer):

        print('deleting', json_object["config"]["layers"][layer]["name"])


        layer_name = json_object["config"]["layers"][layer-1]["name"]
        json_object["config"]["layers"][layer + 1]["inbound_nodes"][0][0][0] = layer_name
        del json_object["config"]["layers"][layer]

        return json_object

    @staticmethod
    def remove_chosen_conv_block_from_json_string(json_object, layer):

        print('Deleting block associated with', json_object["config"]["layers"][layer]["name"])
        conv_name = json_object["config"]["layers"][layer]["name"]
        for i in range(3):
            if (json_object["config"]["layers"][layer]["class_name"] == 'BatchNormalization' or
                    json_object["config"]["layers"][layer]["class_name"] == 'ReLU' or
                    json_object["config"]["layers"][layer]["name"] == conv_name):
                json_object = NNModifier.remove_chosen_layer_from_json_string(json_object, layer)
            else:
                break

        print('Block deleted\n')

        return json_object







