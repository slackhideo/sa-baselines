import os


FILES_BASE_DIR = 'stop_words'


def read_stop_words(filename='stop_words.txt'):
    with open(os.path.join(os.getcwd(), FILES_BASE_DIR, filename), 'r') as f:
        stop_words = f.read().splitlines()

    print(len(stop_words))
    return sorted(list(set(stop_words)))
