from keras import backend as K
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy


def accuracy(y_true, y_pred):
    y_true = y_true[:, :10]
    y_pred = y_pred[:, :10]
    return categorical_accuracy(y_true, y_pred)


def top_5_accuracy(y_true, y_pred):
    y_true = y_true[:, :10]
    y_pred = y_pred[:, :10]
    return top_k_categorical_accuracy(y_true, y_pred)


def categorical_crossentropy_metric(y_true, y_pred):
    y_true = y_true[:, :10]
    y_pred = y_pred[:, :10]
    return categorical_crossentropy(y_true, y_pred)


def soft_categorical_crossentrophy(temperature):
    def soft_categorical_crossentrophy_metric(y_true, y_pred):
        logits = y_true[:, 10:]
        y_soft = K.softmax(logits/temperature)
        y_pred_soft = y_pred[:, 10:]
        return categorical_crossentropy(y_soft, y_pred_soft)
    return soft_categorical_crossentrophy_metric