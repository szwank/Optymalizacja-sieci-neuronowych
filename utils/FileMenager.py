import json
import os
import shutil


class FileManager:

    @staticmethod
    def remove_empty_dir(dir):
        # Pobranie listy dostępnych folderów
        folder_list = os.listdir(path=dir)

        for folder in folder_list:
            if not os.path.isfile(path=dir + '/' + folder):     # Sprawdzenie czy to nie jest plik
                if not os.listdir(path=dir + '/' + folder):     # Sprawdzenie czy folder jest pusty
                    os.rmdir(dir + '/' +folder)  # Usunięcie pustych folderów

    @staticmethod
    def remove_insignificant_dir(dir, number_files_to_be_insignificant=2):
        # Pobranie listy dostępnych folderów
        folder_list = os.listdir(path=dir)

        for folder in folder_list:
            files = os.listdir(path=dir + '/' + folder)
            number_of_files_in_dir = len(files)
            if number_of_files_in_dir <= number_files_to_be_insignificant:
                shutil.rmtree(path=dir + '/' + folder, ignore_errors=True)    # Usunięcie folderów z ilością plików mniejszą,
                                                                        # równą od number_files_to_be_insignificant

    @staticmethod
    def create_folder(path):
        scierzka_zapisu_dir = os.path.join(os.getcwd(), path)
        if not os.path.exists(scierzka_zapisu_dir):  # stworzenie folderu jeżeli nie istnieje
            os.makedirs(scierzka_zapisu_dir)

    @staticmethod
    def get_dictionary_from_json_text_file(path_to_text_file: str):
        file = open(path_to_text_file, "r")
        json_string = file.read()
        layers_accuracy_dict = json.loads(json_string)
        file.close()
        return layers_accuracy_dict