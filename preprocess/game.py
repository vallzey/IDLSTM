# encoding:utf-8
from __future__ import print_function
import pandas as pd
import pickle
import os
import random

BASE_DIR = 'data'
# preprocess/data/game_data_1/user_detail_filter_4.csv
DATA_SOURCE = 'game_data_20'
user_detail_path = os.path.join(BASE_DIR, DATA_SOURCE, 'user_detail_filter_4.csv')
user_item_record = os.path.join(BASE_DIR, DATA_SOURCE, 'user-item.lst')
user_item_delta_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'user-item-delta-time.lst')
user_item_delta_time_neg_record = os.path.join(BASE_DIR, DATA_SOURCE, 'user-item-delta-time-neg.lst')
user_item_continuous_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'user-item-duration.lst')
user_item_accumulate_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'user-item-accumulate-time.lst')
user_item_type_record = os.path.join(BASE_DIR, DATA_SOURCE, 'user-item-type.lst')

index2game_path = os.path.join(BASE_DIR, DATA_SOURCE, 'index2item')
game2index_path = os.path.join(BASE_DIR, DATA_SOURCE, 'item2index')
index2type_path = os.path.join(BASE_DIR, DATA_SOURCE, 'index2type')
type2index_path = os.path.join(BASE_DIR, DATA_SOURCE, 'type2index')


def generate_data():
    out_ui = open(user_item_record, 'w')
    out_uidt = open(user_item_delta_time_record, 'w')
    out_uidtn = open(user_item_delta_time_neg_record, 'w')
    out_uiat = open(user_item_accumulate_time_record, 'w')
    # addc
    out_uict = open(user_item_continuous_time_record, 'w')

    # 添加类别
    out_it = open(user_item_type_record, 'w')

    df = pd.read_csv(user_detail_path,
                     error_bad_lines=False,
                     header=None,
                     names=['timestamp', 'user_id', 'game_name', 'game_type', 'duration'])

    # 如果存在,直接读取
    if os.path.exists(index2game_path) and os.path.exists(game2index_path):
        index2game = pickle.load(open(index2game_path, 'rb'))
        game2index = pickle.load(open(game2index_path, 'rb'))
        print('Total game %d' % len(index2game))
    else:
        print('Build index2game')
        # 将数据按照game分类,在通过记录数量排序
        sorted_series = df.groupby(['game_name']).size().sort_values(ascending=False)
        index2game = sorted_series.keys().tolist()
        print('Most common game is "%s":%d' % (index2game[0], sorted_series[0]))
        print('build game2index')
        # 反向构造game2index的dict
        game2index = dict((v, i) for i, v in enumerate(index2game))
        pickle.dump(index2game, open(index2game_path, 'wb'))
        pickle.dump(game2index, open(game2index_path, 'wb'))

    '''
    type
    '''
    if os.path.exists(index2type_path) and os.path.exists(type2index_path):
        index2type = pickle.load(open(index2type_path, 'rb'))
        type2index = pickle.load(open(type2index_path, 'rb'))
        print('Total type %d' % len(index2type))
    else:
        print('Build index2type')
        # 将数据按照game分类,在通过记录数量排序
        sorted_series = df.groupby(['game_type']).size().sort_values(ascending=False)
        index2type = sorted_series.keys().tolist()
        print('Most common type is "%s":%d' % (index2type[0], sorted_series[0]))
        print('build type2index')
        # 反向构造game2index的dict
        type2index = dict((v, i) for i, v in enumerate(index2type))
        pickle.dump(index2type, open(index2type_path, 'wb'))
        pickle.dump(type2index, open(type2index_path, 'wb'))

    print('start loop')

    count = 0
    user_group = df.groupby(['user_id'])

    # 添加模块,使随机用户
    user_zip = user_group.size().iteritems()
    user_list = list(user_zip)
    random.shuffle(user_list)

    for user_id, length in user_list:
        if count % 10 == 0:
            print("=====count %d======" % count)
        count += 1
        print('%s %d' % (user_id, length))
        # 对没有的用户的游戏事件,通过时间排序
        user_data = user_group.get_group(user_id).sort_values(by='timestamp')
        # 取出游戏名称的序列和时间
        game_seq = user_data['game_name']
        type_seq = user_data['game_type']
        time_seq = user_data['timestamp']
        duration_seq = user_data['duration']
        # 将其中的空值去除
        game_seq = game_seq[game_seq.notnull()]
        type_seq = type_seq[type_seq.notnull()]
        time_seq = time_seq[time_seq.notnull()]
        duration_seq = duration_seq[time_seq.notnull()]
        # diff(-1)表示与下一个时间做对比, *-1是因为这个之是一个负数
        delta_time = pd.to_datetime(time_seq).diff(-1).astype('timedelta64[s]') * - 1
        # apply()表示沿序列应用函数,这里表示将game_seq中的game_name换成game_id
        game_seq = game_seq.apply(lambda x: game2index[x] if pd.notnull(x) else -1).tolist()
        type_seq = type_seq.apply(lambda x: type2index[x] if pd.notnull(x) else -1).tolist()
        delta_time = delta_time.tolist()
        delta_time[-1] = 0

        duration_seq = duration_seq.tolist()

        # 计算真正的delta_time
        '''
        delta_time = [(float(d1) - float(d2)) if (float(d1) - float(d2)) > 0 else 0. for d1, d2 in zip(delta_time, duration_seq)]
        '''
        delta_time_neg = [(float(d1) - float(d2)) for d1, d2 in zip(delta_time, duration_seq)]

        # 用于计算累计量
        time_accumulate = [0]
        for delta in delta_time[:-1]:
            next_time = time_accumulate[-1] + delta
            time_accumulate.append(next_time)

        out_ui.write(str(user_id) + ',')
        out_ui.write(' '.join(str(x) for x in game_seq) + '\n')
        out_it.write(str(user_id) + ',')
        out_it.write(' '.join(str(x) for x in type_seq) + '\n')

        out_uidt.write(str(user_id) + ',')
        out_uidt.write(' '.join(str(x) for x in delta_time) + '\n')
        out_uidtn.write(str(user_id) + ',')
        out_uidtn.write(' '.join(str(x) for x in delta_time_neg) + '\n')
        out_uiat.write(str(user_id) + ',')
        out_uiat.write(' '.join(str(x) for x in time_accumulate) + '\n')
        out_uict.write(str(user_id) + ',')
        out_uict.write(' '.join(str(x) for x in duration_seq) + '\n')

    out_ui.close()
    out_it.close()
    out_uidt.close()
    out_uidtn.close()
    out_uiat.close()
    out_uict.close()


def split_file(split_number, file_path, tr_file_path, te_file_path):
    te_list = []
    with open('data/game_pre/te_user-item.lst', 'r') as te_file:
        for line in te_file:
            te_list.append(line.split(',')[0])

    with open(file_path, 'r') as test_file:
        te_file = open(te_file_path, 'w')
        tr_file = open(tr_file_path, 'w')
        for line in test_file:
            if line.split(',')[0] not in te_list:
                tr_file.write(line)
            else:
                te_file.write(line)


def split_file2(split_number, file_path, tr_file_path, te_file_path):
    with open(file_path, 'r') as test_file:
        te_file = open(te_file_path, 'w')
        tr_file = open(tr_file_path, 'w')
        for i, line in enumerate(test_file):
            if i < split_number:
                tr_file.write(line)
            else:
                te_file.write(line)

def split_all_data(split_number):
    tr_user_item_record = os.path.join(BASE_DIR, DATA_SOURCE, 'tr_user-item.lst')
    te_user_item_record = os.path.join(BASE_DIR, DATA_SOURCE, 'te_user-item.lst')
    # tr_user_item_type_record = os.path.join(BASE_DIR, DATA_SOURCE, 'tr_user-item-type.lst')
    # te_user_item_type_record = os.path.join(BASE_DIR, DATA_SOURCE, 'te_user-item-type.lst')
    tr_user_item_delta_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'tr_user-item-delta-time.lst')
    te_user_item_delta_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'te_user-item-delta-time.lst')
    tr_user_item_delta_time_neg_record = os.path.join(BASE_DIR, DATA_SOURCE, 'tr_user-item-delta-time-neg.lst')
    te_user_item_delta_time_neg_record = os.path.join(BASE_DIR, DATA_SOURCE, 'te_user-item-delta-time-neg.lst')
    tr_user_item_accumulate_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'tr_user-item-accumulate-time.lst')
    te_user_item_accumulate_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'te_user-item-accumulate-time.lst')
    tr_user_item_continuous_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'tr_user-item-duration.lst')
    te_user_item_continuous_time_record = os.path.join(BASE_DIR, DATA_SOURCE, 'te_user-item-duration.lst')
    split_file(split_number, user_item_record, tr_user_item_record, te_user_item_record)
    # split_file(split_number, user_item_record, tr_user_item_type_record, te_user_item_type_record)
    split_file(split_number, user_item_delta_time_record, tr_user_item_delta_time_record,
               te_user_item_delta_time_record)
    split_file(split_number, user_item_delta_time_neg_record, tr_user_item_delta_time_neg_record,
               te_user_item_delta_time_neg_record)
    split_file(split_number, user_item_accumulate_time_record, tr_user_item_accumulate_time_record,
               te_user_item_accumulate_time_record)
    split_file(split_number, user_item_continuous_time_record, tr_user_item_continuous_time_record,
               te_user_item_continuous_time_record)


if __name__ == '__main__':
    # 如果存在,直接读取
    # if os.path.exists(index2game_path) and os.path.exists(user_item_record):
    #     index2game = pickle.load(open(index2game_path, 'rb'))
    #     print('Total game %d' % len(index2game))
    #     user_item_list = pd.read_csv(user_item_record, header=None)
    #     print('Total user %d' % len(user_item_list))
    # exit(0)
    generate_data()
    # user_item_record = os.path.join('','game', 'user-item.lst')
    # f = open(user_item_record, 'w')
    # f.close()
    split_all_data(1800)
