from NNLoader import NNLoader
import h5py

[x_train, x_validation, x_test], [y_train, y_validation, y_test] = NNLoader.load_CIFAR10()
h5f = h5py.File('/home/szwank/Desktop/Optymalizacja-sieci-neuronowych/data/CIFAR10.h5', 'a')
h5f.create_dataset('x_train', data=x_train, dtype='float64')
h5f.create_dataset('x_validation', data=x_validation, dtype='float64')
h5f.create_dataset('x_test', data=x_test, dtype='float64')

h5f.create_dataset('y_train', data=y_train, dtype='float64')
h5f.create_dataset('y_validation', data=y_validation, dtype='float64')
h5f.create_dataset('y_test', data=y_test, dtype='float64')


h5f.close()

