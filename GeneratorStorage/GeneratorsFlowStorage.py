from GeneratorStorage.GeneratorDataLoader import GeneratorDataLoader

class GeneratorsFlowStorage:
    """Class for storage generators flow needed for training neural network."""

    def __init__(self, training_generator, validation_generator,
                 test_generator, generator_data_loader: GeneratorDataLoader):

        self.train_data_generator = training_generator
        self.validation_data_generator = validation_generator
        self.test_data_generator = test_generator

        self. generator_data_loader = generator_data_loader

    def get_train_data_generator_flow(self, batch_size, shuffle):
        return self.generator_data_loader.get_generator_flow(self.train_data_generator, batch_size, shuffle, type_of_generator='train')

    def get_validation_data_generator_flow(self, batch_size, shuffle):
        return self.generator_data_loader.get_generator_flow(self.validation_data_generator, batch_size, shuffle, type_of_generator='validation')

    def get_test_data_generator_flow(self, batch_size, shuffle):
        return self.generator_data_loader.get_generator_flow(self.test_data_generator, batch_size, shuffle, type_of_generator='test')
