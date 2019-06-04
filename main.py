import keras
from keras.models import load_model, save_model
import Train_VGG16_on_CIFAR10
import shallowing_NN
from keras.models import save_model

if __name__ == '__main__':
    # Tworzeie/uczenie sieci
    # Jeżeli model nie istnieje: [], jezeli istnieje podac ścieczke do wag i struktury
    path_to_model = 'Zapis modelu/VGG16-CIFAR10-0.94acc.hdf5'
    if not path_to_model:
        # model = load_model('/home/szwank/Desktop/Optymalizacja-sieci-neuronowych/Zapis modelu/19-04-17 14-35/weights-improvement-98-0.93.hdf5')
        model = Train_VGG16_on_CIFAR10.tran_VGG16_on_CIFAR10()
        Train_VGG16_on_CIFAR10.asses_model_on_CIFAR10(model)

        path_to_model = 'temp/trained_model'
        model.save(path_to_model)
        keras.backend.clear_session()

    # Optymalizacja sieci
    shallowing_NN.assess_conv_layers(path_to_model=path_to_model)
    shallowed_model = shallowing_NN.shallow_network('Zapis modelu/VGG16-CIFAR10-0.94acc.hdf5')
    shallowing_NN.knowledge_distillation(shallowed_model)



