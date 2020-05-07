# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import logging
import argparse
import datetime

import numpy as np
import theano
import theano.tensor as T
import lasagne

from tgate import OutGate, TimeGate
import ConfigParser
import pickle

config = ConfigParser.ConfigParser()
config.read('vmain.cfg')
BASE_DIR = config.get('data_path', 'BASE_DIR')
USER_RECORD_PATH = config.get('data_path', 'USER_RECORD_PATH')
DELTA_TIME_PATH = config.get('data_path', 'DELTA_TIME_PATH')
ACC_TIME_PATH = config.get('data_path', 'ACC_TIME_PATH')

# addv
DURATION_PATH = config.get('data_path', 'DURATION_PATH')

# 下面的两个INDEX好像并没有用到,因为前期的处理已经将ITEM转化成INDEX
INDEX2WORD_PATH = config.get('data_path', 'INDEX2WORD_PATH')
WORD2INDEX_PATH = config.get('data_path', 'WORD2INDEX_PATH')


def load_data(data_attr):
    # max_len = 0,# Max length of setence
    # vocab_size = 0, # vocabulary size
    # debug=False, # return a small set if True
    # val_num=100,  # number of validation sample
    # with_time=False, # return time information
    # with_delta_time=False # return delta time if True else if with_time == True return time
    # return: two dictionary
    # train = {'x':..., 'y':..., 't':...}
    # test = {'x':..., 'y':..., 't':...}
    # v:未做注释,因为还不知道是什么意思
    # max_len 序列长度
    # vocab_size item种类
    max_len = data_attr.get('max_len', 10000)
    vocab_size = data_attr.get('vocab_size', 20000)
    debug = data_attr.get('debug', False)
    # filter = data_attr.get('filter', False)

    with_time = data_attr.get('with_time', False)
    with_delta_time = data_attr.get('with_delta_time', False)
    data_source = data_attr.get('source', 'game')
    data_source = data_source.replace('\r', '')
    # addv
    with_duration = data_attr.get('with_duration', False)

    duration_size = float(data_attr.get('duration_size', 300))
    duration_max_size = float(data_attr.get('duration_max_size', 10000))

    stand = float(data_attr.get('stand', 1))

    # v:可能是如果时间with_time和with_delta_time有一个是true,就表示为true
    with_time = with_time or with_delta_time

    '''v
    # data_source 默认 game
    # prefix : tr_或者te_
    item_seq = [,,,,]
    :return sentences = [item_seq1,item_seq2]
    '''

    def load_file(data_source, prefix, debug=False):
        sentences = []
        user_record_path = os.path.join(BASE_DIR, data_source, prefix + USER_RECORD_PATH)

        # 这里好像的意思是将文件中所有的用户的数据加到一个矩阵中[[]]
        if os.path.exists(user_record_path):
            with open(user_record_path, 'r') as f:
                count = 0
                for line in f:
                    userid, item_seq = line.strip().split(',')
                    item_seq = [int(x) for x in item_seq.split(' ')]
                    # 将用户的全部序列叠加
                    sentences.append(item_seq)
                    count += 1
                    # use a small subset if debug on
                    # 如果是在debug模式,则用50条序列测试
                    if debug and count == 50:
                        break
        else:
            logging.info('{} not exists!'.format(user_record_path))
            exit()

        time_seq = None
        time_file_path = os.path.join(BASE_DIR, data_source, prefix + ACC_TIME_PATH)
        # addv 读取duration
        duration_seq = None
        duration_path = os.path.join(BASE_DIR, data_source, prefix + DURATION_PATH)

        if with_delta_time:
            time_file_path = os.path.join(BASE_DIR, data_source, prefix + DELTA_TIME_PATH)

        if with_time and os.path.exists(time_file_path) and os.path.exists(duration_path):
            time_seq = []
            duration_seq = []

            with open(time_file_path, 'r') as f1, open(duration_path, 'r') as f2:
                count = 0

                for line1, line2 in zip(f1, f2):
                    delta_userid, delta = line1.strip().split(',')
                    duration_userid, duration = line2.strip().split(',')

                    def _(x):
                        try:
                            # 标准化
                            return float(x)
                        except Exception:
                            return 0

                    delta = [float(x) + 0.001 for x in delta.split(' ')]
                    duration = [_(x) / stand for x in duration.split(' ')]

                    if len(delta) != len(sentences[count]) or len(duration) != len(sentences[count]):
                        logging.info('Data conflict at line {}, delete'.format(count))
                        del sentences[count]
                        continue
                    time_seq.append(delta)
                    duration_seq.append(duration)
                    count += 1
                    if debug and count == 50:
                        break
        elif with_time:
            logging.info('Time record not found')
            exit()
        elif with_duration:
            logging.info('duration record not found')
            exit()

        return sentences, time_seq, duration_seq  # addv duration_seq

    '''v
    vocab_size:默认为20000,
    单条用户物品长度(one-hot)不超过vocab_size
    '''

    # addv changing
    def remove_large_word(sentences, vocab_size, time_seq=None, duration_seq=None):
        # addv
        if time_seq is not None and duration_seq is not None:
            sents_ret = []
            dt_ret = []
            ds_ret = []
            pre_time = 0
            # check whether the word is in vocabulary list.
            # if not, we should add the delta time to the next word
            for sent, delta_time, duration in zip(sentences, time_seq, duration_seq):
                _sent = []
                _dt = []
                _ds = []
                # 取出单词中所有单词标记大于vocab_size的单词,同时delta要加上pre_time
                for word, delta, dura in zip(sent, delta_time, duration):
                    if word < vocab_size:
                        _sent.append(word)
                        _dt.append(pre_time + delta)
                        _ds.append(dura)
                        pre_time = 0
                    else:
                        pre_time += delta
                assert (len(_sent) == len(_dt) and len(_sent) == len(_ds))
                sents_ret.append(_sent)
                dt_ret.append(_dt)
                ds_ret.append(_ds)
            return sents_ret, dt_ret, ds_ret
        # remove the word which is larger than max
        elif duration_seq is not None:
            sents_ret = []
            ds_ret = []
            pre_time = 0
            # check whether the word is in vocabulary list.
            # if not, we should add the delta time to the next word
            for sent, duration in zip(sentences, duration_seq):
                _sent = []
                _ds = []
                for word, dura in zip(sent, duration):
                    if word < vocab_size:
                        _sent.append(word)
                        _ds.append(dura)
                assert (len(_sent) == len(_ds))
                sents_ret.append(_sent)
                ds_ret.append(_ds)
            return sents_ret, None, ds_ret
        else:
            return [filter(lambda word: word < vocab_size, sent) for sent in sentences], None, None

    '''v
    max_len:默认值10000
    '''

    # addv
    def cut_sentences(sentences, max_len, time_seq=None, duration_seq=None):
        # remove the sentences: len < 2 and len > max_len
        dt_ret = None
        ds_ret = None
        if max_len:
            sents_ret = [sent[:max_len] for sent in sentences]
            if time_seq is not None:
                dt_ret = [delta_time[:max_len] for delta_time in time_seq]
            if duration_seq is not None:
                ds_ret = [duration[:max_len] for duration in duration_seq]
        else:
            sents_ret = sentences
            if time_seq is not None:
                dt_ret = time_seq
            if duration_seq is not None:
                ds_ret = duration_seq

        return sents_ret, dt_ret, ds_ret

    # addv
    def find_next_new_item(sentences, time_seq, duration_seq, duration_size, duration_max_size):
        # addv
        if time_seq is not None and duration_seq is not None:
            sents_ret = []
            dt_ret = []
            ds_ret = []
            index = 0
            # check whether the word is in vocabulary list.
            # if not, we should add the delta time to the next word
            for sent, delta_time, duration in zip(sentences, time_seq, duration_seq):
                _sent = []
                _dt = []
                _ds = []
                index += 1
                print(index, end=', ')

                '''
                从尾到头
                [:1:-1]
                
                '''
                for _i in range(len(sent))[100::]:
                    # sent[_i] not in sent[0:_i] and
                    # if _i == 110:
                    if sent[_i] not in sent[0:_i] and duration_max_size / stand > duration[_i] > duration_size / stand \
                            and delta_time[_i] > 0:
                        _sent = sent[0:_i + 1]
                        _dt = delta_time[0:_i + 1]
                        _ds = duration[0:_i + 1]

                        sents_ret.append(_sent)
                        dt_ret.append(_dt)
                        ds_ret.append(_ds)
                        print('{}, {}'.format(_i, sent[_i]), end='')
                        break
                print()
            return sents_ret, dt_ret, ds_ret
            # addv
        # elif duration_seq is not None:
        #     sents_ret = []
        #     ds_ret = []
        #     # check whether the word is in vocabulary list.
        #     # if not, we should add the delta time to the next word
        #     for sent, duration in zip(sentences, duration_seq):
        #         _sent = []
        #         _ds = []
        #         for _i in range(len(sent))[50::1]:
        #             # sent[_i] not in sent[0:_i]
        #             if sent[_i] not in sent[0:_i] and duration_max_size / stand > duration[_i] > duration_size / stand:
        #                 _sent = sent[0:_i + 1]
        #                 _ds = duration[0:_i + 1]
        #
        #                 sents_ret.append(_sent)
        #                 ds_ret.append(_ds)
        #                 break
        #     return sents_ret, None, ds_ret
        # else:
        #     sents_ret = []
        #     # check whether the word is in vocabulary list.
        #     # if not, we should add the delta time to the next word  [:1:-1] [800::1]
        #     for sent in sentences:
        #         _sent = []
        #         for _i in range(len(sent))[:1:-1]:
        #             if sent[_i] not in sent[0:_i]:
        #                 _sent = sent[0:_i+1]
        #
        #                 sents_ret.append(_sent)
        #                 break
        #     return sents_ret, None, None
        # logging.error("运行错误")
        exit(0)

    '''
    single_y:true 表示只取最后一个作为测试
    single_y:false 表示每一个对应的y都是x+1
    '''

    # addv
    def generate_x_y(sentences, time_seq=None, duration_seq=None, single_y=True):
        if single_y:
            x = [sent[:-1] for sent in sentences]
            y = [sent[-1] for sent in sentences]
        else:
            x = [sent[:-1] for sent in sentences]
            y = [sent[1:] for sent in sentences]
        t = None
        d = None
        if time_seq is not None:
            t = [delta_time[:-1] for delta_time in time_seq]
        if duration_seq is not None:
            # addv
            d = [duration[:-1] for duration in duration_seq]

        return x, y, t, d

    # addv train_duration_seq
    test_data, test_time_seq, test_duration_seq = load_file(data_source, 'te_', debug)

    logging.info('Remove large word')

    # add
    if vocab_size:
        test_data, test_time_seq, test_duration_seq = remove_large_word(test_data, vocab_size, test_time_seq,
                                                                        test_duration_seq)

    # addv 取出负面的例子
    def remove_negative_word(sentences, duration_size, duration_max_size, time_seq=None, duration_seq=None):
        # addv

        if time_seq is not None and duration_seq is not None:
            sents_ret = []
            dt_ret = []
            ds_ret = []
            pre_time = 0
            # check whether the word is in vocabulary list.
            # if not, we should add the delta time to the next word
            for sent, delta_time, duration in zip(sentences, time_seq, duration_seq):
                _sent = []
                _dt = []
                _ds = []
                # 取出单词中所有单词标记大于vocab_size的单词,同时delta要加上pre_time
                for word, delta, dura in zip(sent, delta_time, duration):
                    if duration_size / stand < dura < duration_max_size / stand:
                        _sent.append(word)
                        _dt.append(pre_time + delta)
                        _ds.append(dura)
                        pre_time = 0
                    else:
                        pre_time += delta
                        print("副样本:{}".format(dura))
                assert (len(_sent) == len(_dt) and len(_sent) == len(_ds))
                sents_ret.append(_sent)
                dt_ret.append(_dt)
                ds_ret.append(_ds)
            logging.info('duration_size:{}'.format(duration_size))
            return sents_ret, dt_ret, ds_ret
            # addv
        if duration_seq is not None:
            sents_ret = []
            ds_ret = []
            pre_time = 0
            # check whether the word is in vocabulary list.
            # if not, we should add the delta time to the next word
            for sent, duration in zip(sentences, duration_seq):
                _sent = []
                _ds = []
                # 取出单词中所有单词标记大于vocab_size的单词,同时delta要加上pre_time
                for word, dura in zip(sent, duration):
                    if duration_size / stand < dura < duration_max_size / stand:
                        _sent.append(word)
                        _ds.append(dura)
                    else:
                        print("副样本:{}".format(dura))
                assert (len(_sent) == len(_ds))
                sents_ret.append(_sent)
                ds_ret.append(_ds)
            return sents_ret, None, ds_ret

    logging.info('find new item')
    next_new_item = True
    if next_new_item:
        test_data, test_time_seq, test_duration_seq = find_next_new_item(test_data, test_time_seq, test_duration_seq,
                                                                         duration_size, duration_max_size)

    # addv 只用测试集合 ***参数 时间 300
    # if with_delta_time:
    #     test_data, test_time_seq, test_duration_seq = remove_negative_word(test_data, duration_size, duration_max_size,
    #                                                                        test_time_seq,
    #                                                                        test_duration_seq)

    # We need test data has more history informatiion
    test_data = filter(lambda sent: len(sent) > 2, test_data)
    if with_time:
        test_time_seq = filter(lambda delta_time: len(delta_time) > 2, test_time_seq)

    # addv
    if with_duration:
        test_duration_seq = filter(lambda duration: len(duration) > 2, test_duration_seq)

    logging.info('cut sentences')
    test_data, test_time_seq, test_duration_seq = cut_sentences(test_data, max_len, test_time_seq, test_duration_seq)

    '''v
    xtr=[[],[]]
    ytr=[[],[]]
    xte=[[],[]]
    yte=[]
    '''
    xte, yte, tte, dte = generate_x_y(test_data, test_time_seq, test_duration_seq)

    logging.info('Test data:{}'.format(len(xte)))
    test = {'x': xte, 'y': yte}
    if with_time:
        test['t'] = tte
        if with_duration:
            test['d'] = dte
    # 将训练集和测试集最后以集合的形式输出
    return test
