from keras.layers import Input, Dense, MaxPool2D, Conv2D, Flatten, Dropout, Activation, BatchNormalization, ReLU, Softmax
from keras.models import Model
from keras import regularizers
from keras import backend as K
import tensorflow as tf
class CreateNN:

    @staticmethod
    def create_VGG16_for_CIFAR10():
        inputs = Input(shape=(32, 32, 3))

        x = Conv2D(64, (3, 3), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = MaxPool2D(pool_size=(2, 2))(x)

        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = MaxPool2D(pool_size=(2, 2))(x)

        x = Conv2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = MaxPool2D(pool_size=(2, 2))(x)

        x = Conv2D(512, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(512, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(512, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = MaxPool2D(pool_size=(2, 2))(x)

        x = Conv2D(512, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(512, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(512, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Flatten()(x)

        x = Dense(10)(x)
        x = BatchNormalization()(x)
        x = Softmax()(x)

        model = Model(inputs=inputs, outputs=x)
        return model

    @staticmethod
    def create_VGG16_for_CIFAR10_v2():
        inputs = Input(shape=(32, 32, 3))

        x = Conv2D(64, (3, 3), padding='same')(inputs)
        # x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        # x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

        x = Conv2D(128, (3, 3), padding='same')(x)
        # x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(128, (3, 3), padding='same')(x)
        # x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

        x = Conv2D(256, (3, 3), padding='same')(x)
        # x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(256, (3, 3), padding='same')(x)
        # x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(256, (3, 3), padding='same')(x)
        # x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

        x = Conv2D(512, (3, 3), padding='same')(x)
        # x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(512, (3, 3), padding='same')(x)
        # x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(512, (3, 3), padding='same')(x)
        # x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

        x = Conv2D(512, (3, 3), padding='same')(x)
        # x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(512, (3, 3), padding='same')(x)
        # x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(512, (3, 3), padding='same')(x)
        # x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Flatten()(x)
        x = Dense(512)(x)
        x = Dense(10)(x)
        # x = BatchNormalization()(x)
        x = Softmax()(x)

        model = Model(inputs=inputs, outputs=x)
        return model

    @staticmethod
    def soft_softmax_layer(T=15):
        def soft_softmax(args):
            denominator = K.exp((args - K.max(args, axis=1, keepdims=True)) / T)   # przeskalowanie zapobiega błędom obliczeniowym
            divider = K.sum(denominator, axis=1, keepdims=True)
            soft_max_output = denominator / divider

            # def grad(dy):
            #      return dy * T**2 * (soft_max_output * (1-soft_max_output))
            return soft_max_output

        return soft_softmax


    @staticmethod
    def get_shallowed_model(weight_decay=0.0001):
        inputs = Input(shape=(32, 32, 3))

        x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = MaxPool2D(pool_size=(2, 2))(x)

        # x = Dropout(0.4)(x)

        x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = MaxPool2D(pool_size=(2, 2))(x)

        x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = MaxPool2D(pool_size=(2, 2))(x)

        x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = MaxPool2D(pool_size=(2, 2))(x)

        x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Flatten()(x)

        x = Dense(10)(x)
        x = BatchNormalization()(x)
        x = Softmax()(x)

        model = Model(inputs=inputs, outputs=x)
        return model