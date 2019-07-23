from shallowing_NN_v2 import assesing_conv_filters, assesing_conv_layers, shallow_network, check_integrity_of_score_file, knowledge_distillation
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model, save_model
import keras.backend as K
from NNModifier import NNModifier
from GeneratorStorage.GeneratorsFlowStorage import GeneratorsFlowStorage
from GeneratorStorage.GeneratorDataLoaderFromDisc import GeneratorDataLoaderFromDisc
from NNHasher import NNHasher
from NNSaver import NNSaver
import os

def get_generators_for_training():
    train_data_generator = ImageDataGenerator(
        featurewise_std_normalization=True,
        featurewise_center=True,
        rotation_range=360,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=[0.9, 1.1],
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        rescale=1.0 / 255)

    validation_data_generator = ImageDataGenerator(
        featurewise_std_normalization=True,
        featurewise_center=True,
        fill_mode='nearest',
        rescale=1.0 / 255)

    test_data_generator = ImageDataGenerator(
        featurewise_std_normalization=True,
        featurewise_center=True,
        fill_mode='nearest',
        rescale=1.0 / 255)

    mean_data_generator = ImageDataGenerator(
        fill_mode='nearest',
        rescale=1.0 / 255)

    meangenerator = mean_data_generator.flow_from_directory('data/' + str(zbior) + '/valid',
                                                            class_mode='binary',
                                                            classes=['ben', 'mal'],
                                                            target_size=(224, 224),
                                                            batch_size=200,
                                                            shuffle=False)
    data = meangenerator[0]
    train_data_generator.fit(data[0])
    validation_data_generator.fit(data[0])
    test_data_generator.fit(data[0])

    data_loader = GeneratorDataLoaderFromDisc(path_to_training_data='data/' + str(zbior) + '/train',
                                              path_to_validation_data='data/' + str(zbior) + '/valid',
                                              path_to_test_data='data/' + str(zbior) + '/test',
                                              classes=['ben', 'mal'],
                                              target_size=(224, 224),
                                              class_mode='binary'
                                              )

    generators_for_training = GeneratorsFlowStorage(training_generator=train_data_generator,
                                                    validation_generator=validation_data_generator,
                                                    test_generator=test_data_generator,
                                                    generator_data_loader=data_loader)
    return generators_for_training




train_whole_layers = False
train_filters = False
optimize_networks = True


if train_whole_layers is True:
    for zbior in range(1, 5):
        path = os.path.join('NetworkA', 'fold' + str(zbior))
        network_name = os.listdir(path)[0]
        path_to_model = os.path.join(path, network_name)



        generators_for_training = get_generators_for_training()


        assesing_conv_layers(path_to_model,
                              generators_for_training=generators_for_training,
                              size_of_clasificator=(100, 100, 1),
                              BATCH_SIZE=64,
                              resume_testing=False,
                             start_from_conv_layer=1)




if train_filters is True:
    for zbior in range(1, 5):
        path = os.path.join('NetworkA', 'fold' + str(zbior))
        network_name = os.listdir(path)[0]
        path_to_model = os.path.join(path, network_name)



        generators_for_training = get_generators_for_training()

        assesing_conv_filters(path_to_model,
                              generators_for_training=generators_for_training,
                              size_of_clasificator=(100, 100, 1),
                              BATCH_SIZE=32,
                              clasificators_trained_at_one_time=32,
                              filters_in_grup_after_division=1,
                              resume_testing=False)

if optimize_networks is True:
    for zbior in range(1, 5):
        path = os.path.join('NetworkA', 'fold' + str(zbior))
        network_name = os.listdir(path)[0]
        path_to_model = os.path.join(path, network_name)

        model = load_model(path_to_model)
        model_dict = NNModifier.convert_model_to_dictionary(model)
        model_hash = NNHasher.hash_model(model)
        K.clear_session()

        path_to_assesing_file_full_layers = os.path.join('NetworkA', model_hash)
        path_to_assesing_file_single_filters = os.path.join('NetworkA', model_hash + 'v2')

        integrity_result = check_integrity_of_score_file(path_to_assesing_file_single_filters, model_dict)

        if integrity_result:
            print("Score file for single filters is correct.")
        else:
            raise(ValueError("File {} have a bug. Result in layers {} aren't correct".format(
                path_to_assesing_file_single_filters, integrity_result)))

        shallow_model = shallow_network(path_to_model, path_to_assesing_file_single_filters, path_to_assesing_file_full_layers)

        path_to_shallowed_model = 'temp/shallowed_model.hdf5'

        save_model(shallow_model, path_to_shallowed_model)

        K.clear_session()

        generators_for_training = get_generators_for_training()

        shallowed_model = knowledge_distillation(path_to_shallowed_model, path_to_model, generators_for_training)

        path_to_shallowed_model = os.path.join(path, 'shallowed_model')
        NNSaver.save_model(shallowed_model, path_to_shallowed_model)