import tensorflow.keras as keras
from keras.layers import Input, Dense, MaxPool2D, Conv2D, Flatten, Dropout, Activation, Softmax
from keras.models import Model
from keras.utils import plot_model
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
        new_model = Model(model.input, model.layers[layer].output)
        return new_model

    @staticmethod
    def add_classifier_to_end(model, number_of_neurons=1000, number_of_classes=10):

        y = model.output
        y = Flatten()(y)
        y = Dense(number_of_neurons)(y)
        y = Dense(number_of_classes)(y)
        y = Softmax()(y)

        return Model(model.input, y)

    @staticmethod
    def remove_layers(model, layers_to_remove):
        # Rozbicie sieci na warstey
        layers = [l for l in model.layers]

        input = layers[0].output
        x = input

        for i in range(1, len(layers)):

            if ~(i in layers_to_remove):  # Jeżeli dana warstwa jest na liście to nie dodawaj jej do sieci
                x = layers[i](x)  # Łączenie warstw sieci neuronowej

        new_model = Model(model.input, x)

        return new_model