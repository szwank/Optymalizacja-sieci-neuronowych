from keras.models import load_model, save_model
from keras.optimizers import SGD
from NNHasher import NNHasher
from NNModifier import NNModifier
from NNSaver import NNSaver
from keras import backend as K
from shallowing_NN import check_integrity_of_score_file, shallow_network_based_on_filters, knowledge_distillation, \
    shallow_network_based_on_whole_layers_remove_random_filters, shallow_network_based_on_whole_layers, \
    assessing_conv_layers
from assesing_cancer_tisue_networks import get_list_of_files_in_directory
from utils.FileMenager import FileManager
from keras.metrics import AUC, accuracy
import os
import time
from GeneratorStorage.GeneratorDataLoaderFromDisc import GeneratorDataLoaderFromDisc
from GeneratorStorage.GeneratorsFlowStorage import GeneratorsFlowStorage
from NNLoader import NNLoader
from keras.preprocessing.image import ImageDataGenerator

def get_generators_for_training(batch_size=16):
    dataset_location = '/media/dysk/datasets/isic_challenge_2017/'

    train_data_gen = ImageDataGenerator(
        featurewise_std_normalization=True,
        featurewise_center=True,
        zoom_range=[0.9, 1.1],
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        rescale=1.0 / 255)

    valid_data_gen = ImageDataGenerator(
        featurewise_std_normalization=True,
        featurewise_center=True,
        fill_mode='nearest',
        rescale=1.0 / 255)
    mean_data_gen = ImageDataGenerator(
        fill_mode='nearest',
        rescale=1.0 / 255)

    mean_generator = iter(
        mean_data_gen.flow_from_directory(os.path.join(dataset_location, 'valid'), class_mode='binary',
                                          classes=['ben', 'mal'], target_size=(224, 224), batch_size=150,
                                          shuffle=False))
    train_data_gen.fit(mean_generator.next()[0])
    valid_data_gen.fit(mean_generator.next()[0])
    train_generator = train_data_gen.flow_from_directory(os.path.join(dataset_location, 'train'), class_mode='binary',
                                                         classes=['ben', 'mal'], target_size=(224, 224),
                                                         batch_size=batch_size, shuffle=True)
    valid_generator = valid_data_gen.flow_from_directory(os.path.join(dataset_location, 'valid'), class_mode='binary',
                                                         classes=['ben', 'mal'], target_size=(224, 224),
                                                         batch_size=batch_size, shuffle=False)

    data_loader = GeneratorDataLoaderFromDisc(path_to_training_data=os.path.join(dataset_location, 'test'),
                                              path_to_validation_data=os.path.join(dataset_location, 'valid'),
                                              path_to_test_data=os.path.join(dataset_location, 'test'),
                                              class_mode='binary',
                                              classes=['ben', 'mal'],
                                              target_size=(224, 224)
                                              )

    generators_for_training = GeneratorsFlowStorage(training_generator=train_data_gen,
                                                    validation_generator=valid_data_gen,
                                                    test_generator=valid_data_gen,
                                                    generator_data_loader=data_loader)
    return generators_for_training

def test_model(model, generator):
    optimizer_SGD = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer_SGD, loss='binary_crossentropy', metrics=['accuracy'])
    test_generator = generator.get_test_data_generator_flow(batch_size=128, shuffle=True)
    scores = model.evaluate_generator(test_generator, steps=len(test_generator))
    return scores


def main():
    models_save_directory = 'Output/models'
    remove_all_if_bellow = [0.0075, 0.015, 0.03, 0.075]
    increase_by = [0.015, 0.075, 0.15, 0.3]

    my_optymalization = True

    path_to_model = '/media/dysk/models/epoch_25_val_loss_0.3064_val_acc_0.860'

    model = load_model(path_to_model)
    model_hash = NNHasher.hash_model(model)
    path_to_score_file = os.path.join('Output/scores', model_hash)
    K.clear_session()



    if my_optymalization is True:
        for remove_if_bellow in remove_all_if_bellow:
            for increase_by_ in increase_by:
                generators_for_training = get_generators_for_training()

                assessing_conv_layers(path_to_model,
                                      generators_for_training=generators_for_training,
                                      size_of_clasificator=(100, 100),
                                      batch_size=16,
                                      )

                FileManager.create_folder('temp')
                path_to_shallowed_model = 'temp/shallowed_model.hdf5'

                K.clear_session()

                shallow_model = shallow_network_based_on_whole_layers_remove_random_filters(
                    path_to_original_model=path_to_model,
                    path_to_assessing_data_full_layers=path_to_score_file
                    )

                NNSaver.save_model(shallow_model, path_to_shallowed_model)

                generators_for_training = get_generators_for_training()

                shallowed_model = knowledge_distillation(path_to_shallowed_model, path_to_model,
                                                         generators_for_training,
                                                         temperature=5)

                scores = test_model(shallowed_model, generators_for_training)

                directory_name = "".join(
                    ['number_of_parameters_', str(shallowed_model.count_params()),
                     '_remove_if_bellow_', str(remove_if_bellow), '_leave_if_above_',
                     str(remove_if_bellow + increase_by_)])

                shallowed_model_name = "".join(['Shallowed_model_test_accuracy_', str(scores[1]), '.hdf5'])

                path_to_shallowed_model = os.path.join(models_save_directory, 'shallowed_model_removing_filters', directory_name,
                                                       shallowed_model_name)
                NNSaver.save_model(shallowed_model, path_to_shallowed_model)

                K.clear_session()

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('Czas wykonywania: {}'.format(end_time - start_time))