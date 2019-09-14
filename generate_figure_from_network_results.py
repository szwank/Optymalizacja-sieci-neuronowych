import json
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model


def round_number_string(input: str, number_after_comma: int):
    try:
        number = float(input)
        number = round(number, number_after_comma)
        return str(number)
    except:
        return input


def change_on_percents(input: str):
    try:
        number = float(input) * 100
        return str(number)
    except:
        return input



def format_category(category: str):
    splitted_category = category.split('/')
    for i, string in enumerate(splitted_category):
        splitted_category[i] = round_number_string(string, 4)
        splitted_category[i] = change_on_percents(splitted_category[i])

    return "/".join(splitted_category)


save_path = '/home/szwank/Desktop/magisterka- zdjęcia'

# original_number_of_params = 30666629
original_number_of_params = 14752114

file = open('optymalization_results-cifar10.txt')

optymalization_scores = file.read()
file.close()
optymalization_scores = json.loads(optymalization_scores)

file = open('scores_of_cancerous_tissue_network.txt')
network_scores = file.read()
file.close()
network_scores = json.loads(network_scores)





for fold in sorted(optymalization_scores):

    for type_of_optymalization in sorted(optymalization_scores[fold]):
        data = list(optymalization_scores[fold][type_of_optymalization].items())

        markers = iter(['o', "^", "s", "D", "X", "X", "X", "X", "X", "X", "X"])
        marker = markers.__next__()
        # Create plot
        fig = plt.figure(figsize=(10, 7))
        ax = plt.subplot(111)
        last = '0.0075'
        for element in sorted(data, key=lambda x: x[0].split('/')[0] if x[0].split('/')[1] != 'None' else 'None' + x[0].split('/')[0]):
            category = element[0]

            if category.split('/')[0] != last:
                marker = markers.__next__()
                last = category.split('/')[0]
            y = element[1]['test_accuracy']
            x = element[1]['number_of_parameters']

            ax.scatter(x=x/original_number_of_params*100, y=y*100, marker=marker, alpha=0.8, edgecolors='none', s=150, label=format_category(category))

        plt.plot()
        if fold != '0':     # for cancer network
            zbior = str(int(fold)-1)
            max_accuracy = max(network_scores[zbior]['accuracy'])
            mean_accuracy = np.mean(network_scores[zbior]['accuracy'])

            plt.hlines(max_accuracy*100, xmin=0, xmax=100, color='red')
            plt.hlines(mean_accuracy*100, xmin=0, xmax=100, color='blue')
            if type_of_optymalization == "shallowed_model_removing_whole_layers":
                name = 'Optymalizacja za pomocą algorytmu Shalowing Network'
            elif type_of_optymalization == "shallowed_model_removing_random_filters":
                name = 'Optymalizacja za pomocą autorskiego algorytmu(losowe filry)'
            else:
                name = 'Optymalizacja za pomocą autorskiego algorytmu'

            title = name + '- podzbiór ' + str(fold)
            plt.title(title)

        else:       # for Cifar 10
            if type_of_optymalization == "shallowed_model_removing_whole_layers":
                name = 'Optymalizacja za pomocą algorytmu Shalowing Network'
            elif type_of_optymalization == "shallowed_model_removing_random_filters":
                name = 'Optymalizacja za pomocą autorskiego algorytmu(losowe filry)'
            else:
                name = 'Optymalizacja za pomocą autorskiego algorytmu'

            title = name + '- zbiór CIFAR-10'
            plt.ylim([99, 100.05])

        plt.title(title)
        plt.legend(loc=4)
        plt.xlim([0, 100])
        plt.gca().minorticks_on()
        plt.grid(which='both')
        plt.xlabel('Stosunek ilości parametrów zoptymalizowanej sieci do ilości parametrów sieci oryginalnej [%]')
        plt.ylabel('Dokładność klasyfikacij na zbiorze testowym [%]')
        # plt.show()
        file_name = title.replace(' ', '_')
        file_name = file_name.replace('ó', 'o')
        file_name = file_name.replace('ą', 'a')
        plt.savefig(os.path.join(save_path, file_name))


