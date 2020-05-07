# encoding:utf-8
import os
import pandas as pd

# 稀疏度
sparse = 1
# 文件地址
pre_game_dirpath = os.path.join('data', 'game')

# 测试文件
te_user_path = os.path.join(pre_game_dirpath, 'te_user-item.lst')
te_user_time_path = os.path.join(pre_game_dirpath, 'te_user-item-delta-time.lst')
te_user_duration_path = os.path.join(pre_game_dirpath, 'te_user-item-duration.lst')
predict_list_path = os.path.join(pre_game_dirpath, 'predict_list')

# 训练文件
tr_user_path = os.path.join(pre_game_dirpath, 'tr_user-item.lst')
tr_user_time_path = os.path.join(pre_game_dirpath, 'tr_user-item-delta-time.lst')
tr_user_duration_path = os.path.join(pre_game_dirpath, 'tr_user-item-duration.lst')

# 新的文件地址
new_user_dirpath = os.path.join('data', 'game_data_1')

# 测试文件
new_te_user_path = os.path.join(new_user_dirpath, 'te_user-item.lst')
new_te_user_time_path = os.path.join(new_user_dirpath, 'te_user-item-delta-time.lst')
new_te_user_duration_path = os.path.join(new_user_dirpath, 'te_user-item-duration.lst')
# 训练文件
new_tr_user_path = os.path.join(new_user_dirpath, 'tr_user-item.lst')
new_tr_user_time_path = os.path.join(new_user_dirpath, 'tr_user-item-delta-time.lst')
new_tr_user_duration_path = os.path.join(new_user_dirpath, 'tr_user-item-duration.lst')

if not os.path.exists(new_user_dirpath):
    os.mkdir(new_user_dirpath)


def init_test():
    def _f(line, index, file_w):
        _ = line.split(',')
        user_id = _[0]
        items = _[1].split()[:index + 1]
        items = " ".join(str(i) for i in items)
        file_w.write('{},{}\n'.format(user_id, items))

    predict_list = pd.read_csv(predict_list_path, header=None, names=['sen_id', 'index', 'item_id'])
    with open(te_user_path, 'r') as file_r1, open(te_user_time_path, 'r') as file_r2, \
            open(te_user_duration_path, 'r') as file_r3, open(new_te_user_path, 'w') as file_w1, \
            open(new_te_user_time_path, 'w') as file_w2, open(new_te_user_duration_path, 'w') as file_w3:
        for line1, line2, line3, index in zip(file_r1, file_r2, file_r3, predict_list['index']):
            _f(line1, index, file_w1)
            _f(line2, index, file_w2)
            _f(line3, index, file_w3)


# init_test()

