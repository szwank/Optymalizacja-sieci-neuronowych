import keras
import Train_VGG16_on_CIFAR10
import shallowing_NN
from keras.models import save_model

if __name__ == '__main__':
    path_to_model = []  # zmienić jeżeli wytrenowany model już istnieje
    if path_to_model is []:
        model = Train_VGG16_on_CIFAR10.tran_VGG16_on_CIFAR10()
        Train_VGG16_on_CIFAR10.asses_model_on_CIFAR10(model)

        path_to_model = 'temp/trained_model'
        model.save_model(path_to_model)
        keras.backend.clear_session()

    shallowing_NN.assesing_conv_layers(path_to_model=path_to_model)



