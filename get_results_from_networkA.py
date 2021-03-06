import json
import os
import math

def get_files_in_directory(directory):
    files = []
    for name in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, name)):
            files.append(name)
    return files

def get_folders_in_directory(directory):
    folders = []
    for name in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, name)):
            folders.append(name)
    return folders

def get_numbers_from_string(string, splited_by=''):
    splited_string = string.split(splited_by)
    numbers_in_string = []

    for string in splited_string:
        try:
            numbers_in_string.append(float(string))
        except:
            pass

    return numbers_in_string

def safe_list_gate(list, index):
    try:
        return list[index]
    except IndexError:
        return None



def main():
    results = {}

    for fold in range(1):        # fold
        path = os.path.join('Zapis modelu')
        folders_in_directory = get_folders_in_directory(path)

        type_of_optymalization = {}

        for folder in folders_in_directory:     # filrst directory(type of optymalization)
            if folder not in ['shallowed_model_removing_filters', 'shallowed_model_removing_random_filters', 'shallowed_model_removing_whole_layers']:
                continue

            folders_in_folder = get_folders_in_directory(os.path.join(path, folder))
            type_of_optymalization_results = {}

            for folder_with_files in folders_in_folder:     # second folder parameters of optymalization

                numbers_in_folder_name = get_numbers_from_string(folder_with_files, splited_by="_")
                result = {}
                remove_if_below = safe_list_gate(numbers_in_folder_name, 1)
                leave_if_above = safe_list_gate(numbers_in_folder_name, 2)
                result['number_of_parameters'] = numbers_in_folder_name[0]

                for i, file in enumerate(get_files_in_directory(os.path.join(path, folder, folder_with_files))):      # files

                    accuracy = result.get('test_accuracy', 0)
                    accuracy = i / (i+1) * accuracy + get_numbers_from_string(file[:-5], "_")[0] / (i+1)
                    result['test_accuracy'] = accuracy

                type_of_optymalization_results["".join([str(remove_if_below), '/', str(leave_if_above)])] = result
            type_of_optymalization[folder] = type_of_optymalization_results
        results[fold] = type_of_optymalization

    file = open('optymalization_results-cifar10.txt', 'a+')
    results = json.dumps(results)
    file.write(results)
    file.close()


if __name__ is "__main__":
    main()


