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
    def create_folder(folder_name):
        scierzka_zapisu_dir = os.path.join(os.getcwd(), folder_name)
        if not os.path.exists(scierzka_zapisu_dir):  # stworzenie folderu jeżeli nie istnieje
            os.makedirs(scierzka_zapisu_dir)