import os
import shutil
from os.path import dirname, abspath
def remove_empty_dir(dir):
    # Pobranie listy dostępnych folderów
    folder_list = os.listdir(path=dir)

    for folder in folder_list:
        if not os.path.isfile(path=dir + '/' + folder):     # Sprawdzenie czy to nie jest plik
            if not os.listdir(path=dir + '/' + folder):     # Sprawdzenie czy folder jest pusty
                os.rmdir(dir + '/' +folder)  # Usunięcie pustych folderów

def remove_insignificant_dir(dir, number_files_to_be_insignificant=2):
    # Pobranie listy dostępnych folderów
    folder_list = os.listdir(path=dir)

    for folder in folder_list:
        files = os.listdir(path=dir + '/' + folder)
        number_of_files_in_dir = len(files)
        if number_of_files_in_dir <= number_files_to_be_insignificant:
            shutil.rmtree(path=dir + '/' + folder, ignore_errors=True)    # Usunięcie folderów z ilością plików mniejszą,
                                                                    # równą od number_files_to_be_insignificant


if __name__ == "__main__":
    # Usunięcie pustych folderów w folderze zapis modelu
    dir_to_search = dirname(dirname(abspath(__file__))) + '/Zapis modelu'
    remove_empty_dir(dir_to_search)

    # Usunięcie folderów mniej niż 3 pliki
    # dir_to_search = 'log'
    # remove_insignificant_dir(dir_to_search)







