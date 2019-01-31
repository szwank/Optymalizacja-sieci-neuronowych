from keras.layers import Input, Dense, MaxPool2D, Conv2D, Flatten, Dropout, Activation
from keras.models import Model
class CreateNN:

    @staticmethod
    def create_VGG16_for_CIFAR10():
        a = Input(shape=(32, 32, 3))

        b = Conv2D(64, (3, 3), activation='relu', padding="same")(a)
        b = Conv2D(64, (3, 3), activation='relu', padding="same")(b)
        b = MaxPool2D(pool_size=(2, 2))(b)

        b = Conv2D(128, (3, 3), activation='relu', padding="same")(b)
        b = Conv2D(128, (3, 3), activation='relu', padding="same")(b)
        b = MaxPool2D(pool_size=(2, 2))(b)

        b = Conv2D(256, (3, 3), activation='relu', padding="same")(b)
        b = Conv2D(256, (3, 3), activation='relu', padding="same")(b)
        b = Conv2D(256, (3, 3), activation='relu', padding="same")(b)
        b = MaxPool2D(pool_size=(2, 2))(b)

        b = Conv2D(512, (3, 3), activation='relu', padding="same")(b)
        b = Conv2D(512, (3, 3), activation='relu', padding="same")(b)
        b = Conv2D(512, (3, 3), activation='relu', padding="same")(b)
        b = MaxPool2D(pool_size=(2, 2))(b)

        b = Conv2D(512, (3, 3), activation='relu', padding="same")(b)
        b = Conv2D(512, (3, 3), activation='relu', padding="same")(b)
        b = Conv2D(512, (3, 3), activation='relu', padding="same")(b)
        b = MaxPool2D(pool_size=(2, 2))(b)

        b = Flatten()(b)
        b = Dropout(0.5)(b)
        b = Dense(4096, activation='relu')(b)
        b = Dense(4096, activation='relu')(b)
        b = Dense(10, activation='softmax')(b)

        model = Model(inputs=a, outputs=b)
        return model

