import pickle
import os

BASE_DIR = 'data'
DATA_SOURCE = 'game_init_filter_2'
index2game_path = os.path.join(BASE_DIR, DATA_SOURCE, 'index2item')
index2game = pickle.load(open(index2game_path, 'rb'))
print(index2game[0])
print(len(index2game))
print(index2game[500])
 