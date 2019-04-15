from keras.layers import Input, Dense, MaxPool2D, Conv2D, Flatten, Dropout, Activation, BatchNormalization, ReLU, Softmax
from keras.models import Model
from keras import regularizers
from keras import backend as K
class CreateNN:

    @staticmethod
    def create_VGG16_for_CIFAR10(weight_decay=0.0001):
        inputs = Input(shape=(32, 32, 3))

        x = Conv2D(64, (3, 3), padding='same')(inputs)
        x = ReLU()(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = MaxPool2D(pool_size=(2, 2))(x)

        # x = Dropout(0.4)(x)

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
    def create_VGG16_for_CIFAR10_v2(weight_decay=0.001):
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
    def loss_for_knowledge_distillation(args, alpha=0.1, T=1):
        """Sztuczka! Zamiana obliczania lossu sieci neurnonowej w ostatniej warstwie

        # Argumenty
            args (tensor): wyjście softmax i logów z orginalnej sieci oraz wyjście softmax i logów z uczonej sieci

        # Zwraca (tensor): loss danej sieci
        """

        ground_truth, logits, ground_truth_student, logits_student = args

        first_part = - K.sum(ground_truth * K.log(ground_truth_student + K.epsilon()), axis=1, keepdims=True)

        q = K.exp(logits_student/T) / K.sum(K.exp(logits_student/T + K.epsilon()), axis=1, keepdims=True)
        p = K.exp(logits/T) / K.sum(K.exp(logits/T + K.epsilon()), axis=1, keepdims=True)

        second_part = - alpha * K.sum(p * K.log(q + K.epsilon()), axis=1, keepdims=True)

        return first_part + second_part

