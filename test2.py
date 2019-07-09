from shallowing_NN_v2 import open_text_file
with open_text_file('[[185.03731]]v2-zła', 'r+') as file1:
    a = file1.read()


    file = open_text_file('[[185.03731]]v2-zła', 'r+')

    print(file.read())

file.close()
