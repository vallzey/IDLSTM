# encoding:utf-8
import os
import pandas as pd

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

# 调整稀疏度
sparse = 5
BASE_DIR = 'data'
DATA_SOURCE = 'pre'
BASE_DIR = 'data'
NEW_DATA_SOURCE = 'game_data_{}'.format(sparse)


def get_game_list():
    BASE_DIR = 'data'
    DATA_SOURCE = 'game_init_filter_2'
    user_detail_path = os.path.join(BASE_DIR, DATA_SOURCE, 'user_detail_filter_4.csv')
    df = pd.read_csv(user_detail_path,
                     error_bad_lines=False,
                     header=None,
                     names=['timestamp', 'user_id', 'game_name', 'game_type', 'duration'])
    sorted_series = df.groupby(['game_name']).size().sort_values(ascending=False)
    sorted_series.to_csv(os.path.join(BASE_DIR, DATA_SOURCE, 'game_count.csv'))


# get_game_list()


def filter_data():
    pre_user_detail_path = os.path.join(BASE_DIR, DATA_SOURCE, 'user_detail_filter_4.csv')
    new_user_detail_dirpath = os.path.join(BASE_DIR, NEW_DATA_SOURCE)
    if not os.path.exists(new_user_detail_dirpath):
        os.mkdir(new_user_detail_dirpath)

    new_user_detail_path = os.path.join(new_user_detail_dirpath, 'user_detail_filter_4.csv')

    game_df = pd.read_csv('data/pre/game_count.csv', header=None, names=['game_name', 'count'])
    pre_user_detail_df = pd.read_csv(pre_user_detail_path, error_bad_lines=False, header=None,
                                     names=['timestamp', 'user_id', 'game_name', 'game_type', 'duration'])

    print "处理前:{}".format(len(pre_user_detail_df))

    filter_game_df = game_df[game_df['count'] <= sparse]
    new_user_detail_df = pre_user_detail_df[~pre_user_detail_df['game_name'].isin(filter_game_df['game_name'])]
    new_user_detail_df.to_csv(new_user_detail_path, header=False, index=False)

    print "处理后:{}".format(len(new_user_detail_df))
    print filter_game_df


filter_data()

def calc_sparse():
    new_user_detail_dirpath = os.path.join(BASE_DIR, NEW_DATA_SOURCE)
    new_user_detail_path = os.path.join(new_user_detail_dirpath, 'user_detail_filter_4.csv')
    df = pd.read_csv(new_user_detail_path, error_bad_lines=False, header=None,
                     names=['timestamp', 'user_id', 'game_name', 'game_type', 'duration'])
    sorted_series = df.groupby(['user_id']).size()
    user_count = len(sorted_series)
    sorted_series = df.groupby(['game_name']).size()
    game_count = len(sorted_series)
    drop_dups = df.drop_duplicates(['user_id', 'game_name'])
    sparse_count = len(drop_dups)
    sparse_rate = float(sparse_count) / (game_count * user_count)
    sparse_rate = 1.0 - sparse_rate
    print '{:.4}'.format(sparse_rate)

calc_sparse()
'''
5
dtlstm_emcp 417_0
dtlstm      417_1


'''