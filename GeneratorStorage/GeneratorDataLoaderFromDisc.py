from GeneratorStorage.GeneratorDataLoader import GeneratorDataLoader
import numpy as np

class GeneratorDataLoaderFromDisc(GeneratorDataLoader):

    def __init__(self, path_to_training_data, path_to_validation_data, path_to_test_data, class_mode, classes, target_size, repeat_labesls: int = 1):
        self.path_to_training_data = path_to_training_data
        self.path_to_validation_data = path_to_validation_data
        self.path_to_test_data = path_to_test_data
        self.class_mode = class_mode
        self.classes = classes
        self.target_size = target_size

        self.repeat_labels = repeat_labesls

        self.types_of_generators = {
            'train': self.get_train_flow,
            'validation': self.get_validation_flow,
            'test': self.get_validation_flow,
        }

    def get_generator_flow(self, data_generator, batch_size, shuffle, type_of_generator, *kargs, **kwargs):
        if type_of_generator in self.types_of_generators:
            return self.types_of_generators[type_of_generator](data_generator, batch_size, shuffle)
        else:
            raise ValueError(
                "Type_of_generator argument is incorrect. Correct valuers are 'train', 'validation', 'test'.")

    def get_train_flow(self, data_generator, batch_size, shuffle):
        return data_generator.flow_from_directory(self.path_to_training_data,
                                                                       class_mode=self.class_mode,
                                                                       classes=self.classes,
                                                                       target_size=self.target_size,
                                                                       batch_size=batch_size,
                                                                       shuffle=shuffle)

    def get_validation_flow(self, data_generator, batch_size, shuffle):
        return data_generator.flow_from_directory(self.path_to_validation_data,
                                                                       class_mode=self.class_mode,
                                                                       classes=self.classes,
                                                                       target_size=self.target_size,
                                                                       batch_size=batch_size,
                                                                       shuffle=shuffle)



    def get_test_flow(self, data_generator, batch_size, shuffle):
        retunr = data_generator.flow_from_directory(self.path_to_test_data,
                                                                       class_mode=self.class_mode,
                                                                       classes=self.classes,
                                                                       target_size=self.target_size,
                                                                       batch_size=batch_size,
                                                                       shuffle=shuffle)



