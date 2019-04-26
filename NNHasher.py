from NNLoader import NNLoader
from NNModifier import NNModifier
import xxhash

class NNHasher:

    @staticmethod
    def hash_model(model):
        [x_train, x_validation, x_test], [y_train, y_validation, y_test] = NNLoader.load_CIFAR10()
        model = NNModifier.remove_last_layer(model)
        prediction = model.predict(x_test[:10])
        return xxhash.xxh64(prediction.tostring(), seed=0).hexdigest()


