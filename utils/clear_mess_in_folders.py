from os.path import dirname, abspath

from utils.FileMenager import FileManager

if __name__ == "__main__":
    # Usunięcie pustych folderów w folderze zapis modelu
    dir_to_search = dirname(dirname(abspath(__file__))) + '/Zapis modelu'
    FileManager.remove_empty_dir(dir_to_search)

    # Usunięcie folderów mniej niż 3 pliki
    # dir_to_search = 'log'
    # remove_insignificant_dir(dir_to_search)







