from shallowing_NN_v2 import assesing_conv_layers
import os


for zbior in range(1, 5):
    path = os.path.join('NetworkA', 'fold' + str(zbior))
    network_name = os.listdir(path)[0]
    path_to_model = os.path.join('NetworkA', 'fold' + str(zbior), network_name)

    assesing_conv_layers(path_to_model, BATCH_SIZE=128,
                         clasificators_trained_at_one_time=64,
                         filters_in_grup_after_division=1,
                         resume_testing=False)
