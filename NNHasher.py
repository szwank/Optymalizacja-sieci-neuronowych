from NNLoader import NNLoader
from NNModifier import NNModifier
import numpy as np


class NNHasher:

    @staticmethod
    def hash_model(model):
        # [x_train, x_validation, x_test], [y_train, y_validation, y_test] = NNLoader.load_CIFAR10()
        input_shape = model.input_shape
        input_data = np.ones((1, input_shape[1], input_shape[2], input_shape[3]))
        model = NNModifier.remove_last_layer(model)
        prediction = model.predict(input_data/255.0)

        return np.array2string(np.dot(prediction, prediction.transpose()))

