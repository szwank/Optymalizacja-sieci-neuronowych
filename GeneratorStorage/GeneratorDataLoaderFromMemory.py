from GeneratorStorage.GeneratorDataLoader import GeneratorDataLoader
from TrainingData import TrainingData


class GeneratorDataLoaderFromMemory(GeneratorDataLoader):

    def __init__(self, training_data: TrainingData):
        self.training_data = training_data

        self.types_of_generators = {
            'train': self.get_train_flow,
            'validation': self.get_validation_flow,
            'test': self.get_validation_flow,
        }

    def get_generator_flow(self, data_generator, batch_size, shuffle, type_of_generator, *args, **kwargs):
        if type_of_generator in self.types_of_generators:
            return self.types_of_generators[type_of_generator](data_generator, batch_size, shuffle, *args, **kwargs)
        else:
            raise ValueError(
                "Type_of_generator argument is incorrect. Correct valuers are 'train', 'validation', 'test'.")

    def get_train_flow(self, data_generator, batch_size, shuffle, *args, **kwargs):
        return data_generator.flow(x=self.training_data.get_training_inputs(),
                                   y=self.training_data.get_training_outputs(),
                                   batch_size=batch_size,
                                   shuffle=shuffle, *args, **kwargs)

    def get_validation_flow(self, data_generator, batch_size, shuffle, *args, **kwargs):
        return data_generator.flow(x=self.training_data.get_validation_inputs(),
                                   y=self.training_data.get_validation_outputs(),
                                   batch_size=batch_size,
                                   shuffle=shuffle, *args, **kwargs)

    def get_test_flow(self, data_generator, batch_size, shuffle, *args, **kwargs):
        return data_generator.flow(x=self.training_data.get_test_inputs(),
                                   y=self.training_data.get_test_outputs(),
                                   batch_size=batch_size,
                                   shuffle=shuffle, *args, **kwargs)