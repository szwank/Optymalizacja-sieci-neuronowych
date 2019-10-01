class DataStorage:
    """A class for storage input data or response data for learning neural networks."""
    def __init__(self, training_data, validation_data, test_data):
        self.training_data = training_data
        self.validation_data = validation_data
        self.test_data = test_data

    def get_training_data(self):
        return self.training_data

    def get_validation_data(self):
        return self.validation_data

    def get_test_data(self):
        return self.test_data

