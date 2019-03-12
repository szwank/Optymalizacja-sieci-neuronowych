import json
import os

class NNSaver:

    @staticmethod
    def save_model(model, file_name):
        json_str = model.to_json()
        with open(file_name, 'w') as outfile:
                json.dump(json_str, outfile)

    @staticmethod
    def save_weights(model, file_name, dir='temp'):
        if not os.path.exists(dir):  # Stworzenie folderu jeżeli nie istnieje.
            os.makedirs(dir)

        if file_name in os.listdir(dir + '/'):  # Usunięcie schematu sieci jeżeli istnieje
            os.remove(dir + '/' + file_name)

        model.save_weights(dir + '/' + file_name)

