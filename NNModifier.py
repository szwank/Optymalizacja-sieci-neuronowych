import tensorflow.keras as keras
import tensorflow as tf
from keras.layers import Input, Dense, MaxPool2D, Conv2D, Flatten, Dropout, Activation, Softmax, Lambda
from keras.models import Model
from keras.utils import plot_model
from CreateNN import CreateNN
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
        layers = [l for l in model.layers]
        model.summary()
        input = layers[0].output

        witch_conv = 1  # Licznik warstw konwolucyjnych

        # dla pierwszej warstwy, zamiana input na x
        if type(layers[1]) == Conv2D:  # Sprawdzenie czy warstwa jest konwolucyjna
            if not (witch_conv in layers_to_remove):  # Jeżeli dana warstwa jest na liście to nie dodawaj jej do sieci
                print(layers[1].input)
                print(input.shape)
                config = layers[1].get_config()
                x = layers[1](input)  # Łączenie warstw sieci neuronowej
            witch_conv += 1

        for i in range(2, len(layers)):
            if type(layers[i]) == Conv2D:        # Sprawdzenie czy warstwa jest konwolucyjna
                if not (witch_conv in layers_to_remove):  # Jeżeli dana warstwa jest na liście to nie dodawaj jej do sieci
                    print('i:', i)
                    print(layers[i].input)
                    # print(x.shape)
                    config = layers[i].get_config()
                    x = layers[i](x)  # Łączenie warstw sieci neuronowej
                witch_conv += 1
            else:
                print('i:', i)
                print(layers[i].input)
                # print(input.shape)
                x = layers[i](x)  # Łączenie warstw sieci neuronowej Jeżeli nie jest konwolucyjna

        return Model(inputs=input, outputs=x)

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
















