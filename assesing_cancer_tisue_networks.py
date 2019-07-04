from shallowing_NN_v2 import assesing_conv_layers
from keras.preprocessing.image import ImageDataGenerator
from GeneratorStorage.GeneratorsFlowStorage import GeneratorsFlowStorage
import os

train = True
validation = False


if train is True:
    for zbior in range(1, 5):
        path = os.path.join('NetworkA', 'fold' + str(zbior))
        network_name = os.listdir(path)[0]
        path_to_model = os.path.join(path, network_name)



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

        generators_for_training = GeneratorsFlowStorage(training_generator=train_data_generator, validation_generator=validation_data_generator,
                                                        test_generator=test_data_generator, flow_from_directory=True,
                                                        path_to_train_data='data/' + str(zbior) + '/train',
                                                        path_to_validation_data='data/' + str(zbior) + '/valid',
                                                        path_to_test_data='data/' + str(zbior) + '/test',
                                                        classes=['ben', 'mal'],
                                                        target_size=(224, 224),
                                                        class_mode='binary'
                                                        )


        assesing_conv_layers(path_to_model,
                             generators_for_training=generators_for_training,
                             BATCH_SIZE=64,
                             clasificators_trained_at_one_time=64,
                             filters_in_grup_after_division=1,
                             resume_testing=False)


