import json
from keras.models import model_from_json


class NNLoader:

    @staticmethod
    def load(file_name):
        with open(file_name) as json_file:
            model = json.load(json_file)

        return model_from_json(model)


