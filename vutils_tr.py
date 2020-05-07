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
    with_time = data_attr.get('with_time', False)
    with_delta_time = data_attr.get('with_delta_time', False)
    data_source = data_attr.get('source', 'game')
    data_source = data_source.replace('\r', '')
    # addv
    with_duration = data_attr.get('with_duration', False)

    stand = float(data_attr.get('stand', 1))

    logging.info('Load data using max_len {}, vocab_size {}'.format(max_len, vocab_size))

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

        # addv
        duration_seq = None
        duration_path = os.path.join(BASE_DIR, data_source, prefix + DURATION_PATH)

        if with_delta_time:
            time_file_path = os.path.join(BASE_DIR, data_source, prefix + DELTA_TIME_PATH)

        if with_time and os.path.exists(time_file_path) and os.path.exists(duration_path):

            time_seq = []
            duration_seq = []

            with open(time_file_path, 'r') as f1, open(duration_path, 'r') as f2:
                count = 0

                for line1,line2 in zip(f1,f2):
                    delta_userid, delta = line1.strip().split(',')
                    duration_userid, duration = line2.strip().split(',')

                    def _(x):
                        try:
                            return float(x)
                        except Exception:
                            return 0

                    delta = [float(x)+0.001 for x in delta.split(' ')]
                    duration = [_(x)/stand for x in duration.split(' ')]

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
        elif time_seq is not None:
            sents_ret = []
            dt_ret = []
            pre_time = 0
            # check whether the word is in vocabulary list.
            # if not, we should add the delta time to the next word
            for sent, delta_time in zip(sentences, time_seq):
                _sent = []
                _dt = []
                for word, delta in zip(sent, delta_time):
                    if word < vocab_size:
                        _sent.append(word)
                        _dt.append(pre_time + delta)
                        pre_time = 0
                    else:
                        pre_time += delta
                assert (len(_sent) == len(_dt))
                sents_ret.append(_sent)
                dt_ret.append(_dt)
            return sents_ret, dt_ret, None
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

    '''v
    最后测试数据是否合理
    但是我觉得完全没有比必要
    '''

    def check(sentences, time_seq=None):
        # show the data statics
        logging.info('-------------------------------')
        max_len = 0
        total = 0
        lengths = []
        if time_seq is not None:
            for delta_time, sent in zip(time_seq, sentences):
                assert (len(delta_time) == len(sent))
        for sent in sentences:
            length = len(sent)
            lengths.append(length)
            total += length
            max_len = max_len if max_len > length else length
        if len(lengths) > 0:
            logging.info('-  Sentence number: {}'.format(len(sentences)))
            logging.info('-  Max setence length {}'.format(max_len))
            logging.info('-  average sentence length {}'.format(total * 1. / len(lengths)))
            logging.info('-  90% length {}'.format(sorted(lengths)[int(len(lengths) * 0.9)]))
        logging.info('------------------------------')

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
            d = [duration[:-1] for duration in duration_seq]

        return x, y, t, d

    # addv train_duration_seq
    train_data, train_time_seq, train_duration_seq = load_file(data_source, 'tr_', debug)

    logging.info('Remove large word')

    # add
    if vocab_size:
        train_data, train_time_seq, train_duration_seq = remove_large_word(train_data, vocab_size, train_time_seq,
                                                                           train_duration_seq)

    # addv
    # if with_duration:
    #     train_data, train_time_seq, train_duration_seq = remove_negative_word(train_data, 30, train_time_seq,
    #                                                                           train_duration_seq)
    # addv 只用测试集合

    # remove data which is too short
    train_data = filter(lambda sent: len(sent) > 1, train_data)
    # We need test data has more history informatiion
    if with_time:
        train_time_seq = filter(lambda delta_time: len(delta_time) > 1, train_time_seq)

    # addv
    if with_duration:
        train_duration_seq = filter(lambda duration: len(duration) > 1, train_duration_seq)

    logging.info('cut sentences')
    train_data, train_time_seq, train_duration_seq = cut_sentences(train_data, max_len, train_time_seq,
                                                                   train_duration_seq)
    check(train_data)

    '''v
    xtr=[[],[]]
    ytr=[[],[]]
    xte=[[],[]]
    yte=[]
    '''
    xtr, ytr, ttr, dtr = generate_x_y(train_data, train_time_seq, train_duration_seq, single_y=False)

    logging.info('Train data:{}'.format(len(xtr)))

    train = {'x': xtr, 'y': ytr}
    if with_time:
        train['t'] = ttr
        if with_duration:
            train['d'] = dtr
    # 将训练集和测试集最后以集合的形式输出
    return train
