from TrainingData import TrainingData
from keras.preprocessing.image import ImageDataGenerator

class GeneratorsStorage:
    """Class for storage generators needed for training neural network."""

    def __init__(self, training_generator, validation_generator,
                 test_generator, flow_from_directory, training_data: TrainingData = None,
                 path_to_train_data=None, path_to_validation_data=None, path_to_test_data=None,
                 classes=None, target_size=None, class_mode=None):

        self.train_data_generator = training_generator
        self.validation_data_generator = validation_generator
        self.test_data_generator = test_generator

        self.flow_from_directory = flow_from_directory

        if flow_from_directory is True and None in (path_to_test_data, path_to_train_data, path_to_validation_data):
            raise ValueError("For flow from directory set all 3 paths to data.")
        if flow_from_directory is False and path_to_train_data is None:
            raise ValueError("For flow from loaded data set training data.")

        self.training_data = training_data

        self.path_to_train_data = path_to_train_data
        self.path_to_validation_data = path_to_validation_data
        self.path_to_test_data = path_to_test_data

        self.classes = classes
        self.target_size = target_size
        self.class_mode = class_mode


    def get_train_data_generator_flow(self, batch_size, shuffle):
        if self.flow_from_directory is False:
            return self.train_data_generator.flow(x=self.training_data.get_training_inputs(),
                                                  y=self.training_data.get_training_outputs(),
                                                  batch_size=batch_size,
                                                  shuffle=shuffle)

        else:
            return self.train_data_generator.flow_from_directory(self.path_to_train_data,
                                                                 class_mode=self.class_mode,
                                                                 classes=self.classes,
                                                                 target_size=self.target_size,
                                                                 batch_size=batch_size,
                                                                 shuffle=shuffle)


    def get_validation_data_generator_flow(self, batch_size, shuffle):
        if self.flow_from_directory is False:
            return self.validation_data_generator.flow(x=self.training_data.get_validation_inputs(),
                                                       y=self.training_data.get_validation_outputs(),
                                                       batch_size=batch_size,
                                                       shuffle=shuffle)
        else:
            return self.validation_data_generator.flow_from_directory(self.path_to_validation_data,
                                                                      class_mode=self.class_mode,
                                                                      classes=self.classes,
                                                                      target_size=self.target_size,
                                                                      batch_size=batch_size,
                                                                      shuffle=shuffle)

    def get_test_data_generator_flow(self, batch_size, shuffle):
        if self.flow_from_directory is False:
            return self.test_data_generator.flow(x=self.training_data.get_test_inputs(),
                                                 y=self.training_data.get_test_outputs(),
                                                 batch_size=batch_size,
                                                 shuffle=shuffle)
        else:
            return self.test_data_generator.flow_from_directory(self.path_to_test_data,
                                                                class_mode=self.class_mode,
                                                                classes=self.classes,
                                                                target_size=self.target_size,
                                                                batch_size=batch_size,
                                                                shuffle=shuffle)
