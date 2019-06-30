from DataStorage import DataStorage

class TrainingData:
    """Class for storage training data for training neural networks."""

    def __init__(self, input_data: DataStorage, output_labels: DataStorage):
        self.input_data = input_data
        self.output_labels = output_labels

    def get_training_inputs(self):
        return self.input_data.get_training_data()

    def get_validation_inputs(self):
        return self.input_data.get_validation_data()

    def get_test_inputs(self):
        return self.input_data.get_test_data()

    def get_training_outputs(self):
        return self.output_labels.get_training_data()

    def get_validation_outputs(self):
        return self.output_labels.get_validation_data()

    def get_test_outputs(self):
        return self.output_labels.get_test_data()

