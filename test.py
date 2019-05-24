import keras.backend as K
import tensorflow as tf
import keras
from keras.layers import Input, Conv2D, Dense, Lambda, Flatten, Softmax, BatchNormalization, ReLU, MaxPool2D
from keras.models import Model, load_model
from keras.optimizers import SGD
from NNLoader import NNLoader
from CreateNN import CreateNN


shallowed_model = load_model('Zapis modelu/19-05-06 08-44/weights-improvement-49-2.17.hdf5', compile=False)
# shallowed_model.load_weights(dir_to_original_model, by_name=True)
shallowed_model.layers.pop()
shallowed_model.layers.pop()
shallowed_model.layers.pop()
shallowed_model.layers.pop()
outputs = Softmax()(shallowed_model.layers[-1].output)
shallowed_model = Model(inputs=shallowed_model.inputs, outputs=outputs)
shallowed_model.load_weights('Zapis modelu/19-05-06 08-44/weights-improvement-49-2.17.hdf5', by_name=True)
shallowed_model.summary()
optimizer_SGD = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
shallowed_model.compile(optimizer=optimizer_SGD, loss='categorical_crossentropy', metrics=['accuracy'])


[x_train, x_validation, x_test], [y_train, y_validation, y_test] = NNLoader.load_CIFAR10()

scores = shallowed_model.evaluate(x=x_test/255.0,
                                  y=y_test,
                                  verbose=1,
                                      )
print(scores)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

scores = shallowed_model.evaluate(x=x_train/255.0,
                                  y=y_train,
                                  verbose=1,
                                      )
print(scores)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])