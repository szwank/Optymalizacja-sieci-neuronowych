from keras.layers import Input, Dense, MaxPool2D, Conv2D, Flatten, Dropout, Activation, BatchNormalization, ReLU, Softmax
from keras.models import Model
from keras import regularizers
class CreateNN:

    @staticmethod
    def create_VGG16_for_CIFAR10(weight_decay=0.01):
        inputs = Input(shape=(32, 32, 3))

        x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

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
        x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

        x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

        x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Flatten()(x)
        x = Dense(512)(x)
        x = Dense(10)(x)
        x = BatchNormalization()(x)
        x = Softmax()(x)

        model = Model(inputs=inputs, outputs=x)
        return model

