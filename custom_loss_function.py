import keras.backend as K
import tensorflow as tf
from keras.losses import binary_crossentropy, mean_squared_error, mean_absolute_error
import tensorflow as tf

def categorical_crossentropy_loss(y_true, y_pred):
    return - K.sum(y_true * K.log(y_pred + K.epsilon()), keepdims=True, axis=-1)

def tensorflow_binary_corssentropy(target, output, from_logits=False):
    if not from_logits:
        # transform back to logits
        _epsilon = tf.convert_to_tensor(K.epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.log(output / (1 - output))

    return tf.nn.sigmoid_cross_entropy_with_logits(labels=target,
                                                   logits=output)

def binary_crossentropy_loss(y_true, y_pred):
    return K.mean(tensorflow_binary_corssentropy(y_true, y_pred), axis=-1, keepdims=True)


def categorical_knowledge_distillation_loos(alpha_const, temperature, number_of_outputs_in_last_layer):
    def categorical_knowledge_distillation(y_true, y_pred):
        y_true = y_true[:, :number_of_outputs_in_last_layer]

        y_pred, logits_shallowed_pred, original_logits = y_pred[:, :number_of_outputs_in_last_layer],\
                                                         y_pred[:, number_of_outputs_in_last_layer:number_of_outputs_in_last_layer * 2],\
                                                         y_pred[:, number_of_outputs_in_last_layer * 2:]

        y_soft = K.softmax(original_logits / temperature)

        y_soft_pred = K.softmax(logits_shallowed_pred / temperature)

        return (1-alpha_const) * categorical_crossentropy_loss(y_true, y_pred) + alpha_const * categorical_crossentropy_loss(y_soft, y_soft_pred)
    return categorical_knowledge_distillation


def binary_knowledge_distillation_loos(alpha_const, temperature, number_of_outputs_in_last_layer):
    def binary_knowledge_distillation(y_true, y_pred):

        y_true = y_true[:, :number_of_outputs_in_last_layer]

        y_pred, logits_shallowed_pred, original_logits = y_pred[:, :number_of_outputs_in_last_layer],\
                                                         y_pred[:, number_of_outputs_in_last_layer:number_of_outputs_in_last_layer * 2],\
                                                         y_pred[:, number_of_outputs_in_last_layer * 2:]

        y_soft = K.sigmoid(original_logits / temperature)

        y_soft_pred = K.sigmoid(logits_shallowed_pred / temperature)

        return (1-alpha_const) * binary_crossentropy(y_true, y_pred) + alpha_const * binary_crossentropy(y_soft, y_soft_pred)
    return binary_knowledge_distillation


def loss_of_ground_truth(y_true, y_pred):
    return - K.sum(y_true * K.log(y_pred + K.epsilon()), axis=1, keepdims=True)

def loss_for_many_clasificators(y_true, y_pred):
    loss = K.zeros((1, ), 'float32')
    for i in range(4):
        start_index = i*10
        end_index = (i+1)*10
        loss = K.sum((loss, K.categorical_crossentropy(y_true[start_index:end_index], y_pred[start_index:end_index])))
    return loss
