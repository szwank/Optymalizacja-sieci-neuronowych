from DataGenerator_for_knowledge_distillation import DG_for_kd

params = {'dim': (32, 32),
              'batch_size': 128,
              'n_classes': 10,
              'n_channels': 3,
              'shuffle': True,
              'inputs_number': 3}

training_gen = DG_for_kd(x_data_name='x_train', data_dir='data/CIFAR10.h5',
                         dir_to_weights='Zapis modelu/VGG16-CIFAR10-0.94acc.hdf5', **params)