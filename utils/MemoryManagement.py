from numba import cuda
import keras.backend as K
import tensorflow as tf

class MemoryManagement:

    @staticmethod
    def relase_GPU_memory():
        K.clear_session()
        cuda.select_device(0)
        cuda.close()
        ses = K.get_session()
        config = tf.ConfigProto()
        K.tensorflow_backend.set_session(tf.Session(config=config))