from keras import backend as K
from keras.losses import categorical_crossentropy, mean_squared_error, mean_absolute_error
import tensorflow as tf

def categorical_crossentropy_loss(y_true, y_pred):
    return - K.sum(y_true * K.log(y_pred + K.epsilon()), keepdims=True, axis=-1)


def knowledge_distillation_loos(alpha_const, temperature):
    def knowledge_distillation(y_true, y_pred):
        y_true, logits = y_true[:, :10], y_true[:, 10:]

        y_soft = K.softmax(logits/temperature)

        y_pred, logits_pred = y_pred[:, :10], y_pred[:, 10:]

        y_soft_pred = K.softmax(logits_pred / temperature)

        return categorical_crossentropy_loss(y_true, y_pred) + alpha_const * categorical_crossentropy_loss(y_soft, y_soft_pred)
    return knowledge_distillation


def loss_of_ground_truth(y_true, y_pred):
    return - K.sum(y_true * K.log(y_pred + K.epsilon()), axis=1, keepdims=True)

def loss_for_many_clasificators(y_true, y_pred):
    loss = K.zeros((1, ), 'float32')
    for i in range(4):
        start_index = i*10
        end_index = (i+1)*10
        loss = K.sum((loss, K.categorical_crossentropy(y_true[start_index:end_index], y_pred[start_index:end_index])))
    return loss
