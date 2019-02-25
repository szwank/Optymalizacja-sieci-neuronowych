from keras.utils import plot_model
import os

class Create_NN_graph:

    @staticmethod
    def create_NN_graph(model, name='temp.png', dir='temp'):
        if not os.path.exists(dir):  # Stworzenie folderu jeżeli nie istnieje.
            os.makedirs(dir)
        if name in os.listdir(dir + '/'):  # Usunięcie schematu sieci jeżeli istnieje
            os.remove(dir + '/' + name)
        plot_model(model, to_file=dir + '/' + name, show_shapes=True)