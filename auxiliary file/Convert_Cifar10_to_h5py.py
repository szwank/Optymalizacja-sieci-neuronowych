from NNLoader import NNLoader
import numpy as np
import h5py

[x_train, x_validation, x_test], [y_train, y_validation, y_test] = NNLoader.load_CIFAR10()
h5f = h5py.File('data/CIFAR10.h5', 'w')
h5f.create_dataset('x_train', data=x_train)
h5f.create_dataset('x_validation', data=x_validation)
h5f.create_dataset('x_test', data=x_test)

h5f.create_dataset('y_train', data=y_train)
h5f.create_dataset('y_validation', data=y_validation)
h5f.create_dataset('y_test', data=y_test)


h5f.close()
# partition = dict(train=x_train, validation=x_validation, test=x_test)


a=1
