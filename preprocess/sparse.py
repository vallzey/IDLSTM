import os
import pandas as pd

game_dirpath = os.path.join('data', 'game_20')
te_data = os.path.join(game_dirpath, 'te_user-item.lst')
tr_data = os.path.join(game_dirpath, 'tr_user-item.lst')

count = 0
with open(te_data, 'r') as file:
    for line in file:
        seq = line.split(',')[1].split()
        count += len(seq)

with open(tr_data, 'r') as file:
    for line in file:
        seq = line.split(',')[1].split()
        count += len(seq)

print count