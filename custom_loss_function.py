from keras import backend as K
from keras.losses import categorical_crossentropy, mean_squared_error, mean_absolute_error

def categorical_crossentropy_loss(y_true, y_pred):
    return - K.sum(y_true * K.log(y_pred + K.epsilon()), keepdims=True, axis=-1)


def knowledge_distillation_loos(alpha_const, temperature):
    def knowledge_distillation(y_true, y_pred):
        y_true, logits = y_true[:, :10], y_true[:, 10:]

        # y_soft = K.softmax(logits/temperature)

        y_pred, logits_pred = y_pred[:, :10], y_pred[:, 10:]

        # y_soft_pred = K.softmax(logits_pred / temperature)

        return categorical_crossentropy_loss(y_true, y_pred) + alpha_const * mean_squared_error(logits, logits_pred)
    return knowledge_distillation


def loss_of_ground_truth(y_true, y_pred):
    return - K.sum(y_true * K.log(y_pred + K.epsilon()), axis=1, keepdims=True)
