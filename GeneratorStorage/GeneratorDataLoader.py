

class GeneratorDataLoader:

    def __init__(self):
        raise NotImplementedError()

    def get_generator_flow(self, data_generator, batch_size, shuffle, type_of_generator, *kargs, **kwargs):
        raise NotImplementedError()
