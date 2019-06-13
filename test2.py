from shallowing_NN_v2 import check_integrity_of_score_file,remove_choosen_keys
from keras.models import load_model
from NNHasher import NNHasher
import json

path_to_original_model = 'Zapis modelu/VGG16-CIFAR10-0.94acc.hdf5'

model = load_model(path_to_original_model)
model_hash = NNHasher.hash_model(model)

model_architecture = model.to_json(indent=4)
model_architecture = json.loads(model_architecture)
wrong_scores = check_integrity_of_score_file(str(model_hash) + 'v2', model_architecture)

for i, element in enumerate(wrong_scores):
    wrong_scores[i] = str(element)

file = open(str(model_hash) + 'v2')
dictionary = file.read()
dictionary = json.loads(dictionary)
remove_choosen_keys(dictionary, wrong_scores)
print(dictionary.keys())

