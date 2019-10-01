from keras import backend as K
import tensorflow as tf
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, binary_accuracy


def categorical_accuracy_metric(number_of_outputs_in_last_layer):
    def accuracy_metric(y_true, y_pred):
        y_true = y_true[:, :number_of_outputs_in_last_layer]
        y_pred = y_pred[:, :number_of_outputs_in_last_layer]
        return categorical_accuracy(y_true, y_pred)
    return accuracy_metric


def binary_accuracy_metric(number_of_outputs_in_last_layer):
    def accuracy_metric(y_true, y_pred):
        y_true = y_true[:, :number_of_outputs_in_last_layer]
        y_pred = y_pred[:, :number_of_outputs_in_last_layer]
        return binary_accuracy(y_true, y_pred)
    return accuracy_metric


def top_5_accuracy(y_true, y_pred):
    y_true = y_true[:, :10]
    y_pred = y_pred[:, :10]
    return top_k_categorical_accuracy(y_true, y_pred)


def categorical_crossentropy_metric(number_of_outputs_in_last_layer):
    def categorical_crossentrophy_metric_(y_true, y_pred):
        y_true = y_true[:, :number_of_outputs_in_last_layer]
        y_pred = y_pred[:, :number_of_outputs_in_last_layer]
        return categorical_crossentropy(y_true, y_pred)
    return categorical_crossentrophy_metric_


def binary_crossentropy_metric(number_of_outputs_in_last_layer):
    def binary_crossentrophy_metric_(y_true, y_pred):
        y_true = y_true[:, :number_of_outputs_in_last_layer]
        y_pred = y_pred[:, :number_of_outputs_in_last_layer]
        return binary_crossentropy(y_true, y_pred)
    return binary_crossentrophy_metric_


def soft_categorical_crossentrophy(temperature, number_of_outputs_in_last_layer):
    def soft_categorical_crossentrophy_metric(y_true, y_pred):
        logits_shallowed_pred, original_logits = y_pred[:, number_of_outputs_in_last_layer:number_of_outputs_in_last_layer * 2], \
                                                 y_pred[:, number_of_outputs_in_last_layer * 2:]

        y_soft = K.softmax(original_logits / temperature)
        y_soft_pred = K.softmax(logits_shallowed_pred / temperature)

        return categorical_crossentropy(y_soft, y_soft_pred)

    return soft_categorical_crossentrophy_metric


def soft_binary_crossentrophy(temperature, number_of_outputs_in_last_layer):
    def soft_binary_crossentrophy_metric(y_true, y_pred):
        logits_shallowed_pred, original_logits = y_pred[:, number_of_outputs_in_last_layer:number_of_outputs_in_last_layer * 2], \
                                                 y_pred[:, number_of_outputs_in_last_layer * 2:]

        y_soft = K.sigmoid(original_logits / temperature)
        y_soft_pred = K.sigmoid(logits_shallowed_pred / temperature)

        return binary_crossentropy(y_soft, y_soft_pred)

    return soft_binary_crossentrophy_metric


def mean_accuracy(number_of_inputs, lenght_of_input):
    def mean_accuracy_metric(y_true, y_pred):
        sum_of_accuracy = categorical_accuracy(y_true[0:lenght_of_input], y_pred[0:lenght_of_input])

        for i in range(1, number_of_inputs):
            start_index = i * lenght_of_input
            end_index = (i+1) * lenght_of_input
            accuracy = categorical_accuracy(y_true[start_index: end_index], y_pred[start_index: end_index])
            sum_of_accuracy = sum_of_accuracy[-1] + accuracy[-1]

        return sum_of_accuracy

    return mean_accuracy_metric
