from keras import backend as K
from keras.losses import categorical_crossentropy


def knowledge_distillation_loos(alpha_const, temperature):
    def knowledge_distillation(y_true, y_pred):
        y_true, logits = y_true[:, :10], y_true[:, 10:]

        y_soft = K.softmax(logits/temperature)

        y_pred, y_pred_soft = y_pred[:, :10], y_pred[:, 10:]

        return alpha_const * categorical_crossentropy(y_true, y_pred) + categorical_crossentropy(y_soft, y_pred_soft)
    return knowledge_distillation


def loss_of_ground_truth(y_true, y_pred):
    return - K.sum(y_true * K.log(y_pred + K.epsilon()), axis=1, keepdims=True)


def loss_of_logits(y_true, y_pred):

    alpha = 0.15
    T = 1

    q_denominator = K.exp((y_pred - K.max(y_pred, axis=1, keepdims=True)) / T)
    q_devider = K.sum(q_denominator, axis=1, keepdims=True)
    q = q_denominator / q_devider

    p_denominator = K.exp((y_true - K.max(y_true, axis=1, keepdims=True)) / T)
    p_devider = K.sum(p_denominator, axis=1, keepdims=True)
    p = p_denominator / p_devider

    return - alpha * K.sum(p * K.log(q + K.epsilon()), axis=1, keepdims=True)