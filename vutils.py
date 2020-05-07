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

import random

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


def genPlist(data_size, batch_size):
    plist = range(0, data_size, batch_size)
    random.shuffle(plist)
    return plist


def softmax(x):
    # return softmax on x
    x = np.array(x)
    x /= np.max(x)
    e_x = np.exp(x)
    out = e_x / e_x.sum()
    return out


def sigmoid(x):
    # return sigmoid on x
    x = np.array(x)
    out = 1. / (1 + np.exp(-x))
    return out


# 在训练数据的时候使用,输入的data已经是经过batch处理,
# vocab_size一般是item数量的整数
def prepare_data(data, vocab_size, one_hot=False, sigmoid_on=False, duration_size=300., duration_max_size=10000.,
                 stand=1):
    '''
    convert list of data into numpy.array
    padding 0
    generate mask
    '''
    x_origin = data['x']
    y_origin = data['y']
    t_origin = data.get('t', None)
    d_origin = data.get('d', None)

    # 想使用ndim表示单词的维度,如果单词不是one-hot,就是一维
    ndim = 1 if not one_hot else vocab_size

    # 找出数据集的大小和最大长度
    # 找到批次中最大的数量作为mask的长度
    lengths_x = [len(s) for s in x_origin]
    n_samples = len(x_origin)  # test:20
    max_len = np.max(lengths_x)  # test:5122

    # x (batch,max_len)
    x = np.zeros((n_samples, max_len)).astype('int32')
    t = np.zeros((n_samples, max_len)).astype('float')
    d = np.zeros((n_samples, max_len)).astype('float')
    mask = np.zeros((n_samples, max_len)).astype('float')

    # addv 通过duration筛选出正样本
    pos = np.zeros((n_samples, max_len)).astype('float')

    for idx, sent in enumerate(x_origin):
        x[idx, :lengths_x[idx]] = sent
        mask[idx, :lengths_x[idx]] = 1.
        if t_origin is not None:
            tmp_t = t_origin[idx]
            if sigmoid_on:
                tmp_t = sigmoid(tmp_t)
            t[idx, :int(np.sum(mask[idx]))] = tmp_t
            # addv
            if d_origin is not None:
                tmp_d = d_origin[idx]

                if sigmoid_on:
                    tmp_d = sigmoid(tmp_d)
                d[idx, :int(np.sum(mask[idx]))] = tmp_d

                # addv 通过duration筛选出正样本 ***参数 时间 300
                # tmp_d.append(tmp_d[0])
                # tmp_d = tmp_d[1:len(tmp_d)]
                # print(tmp_d)
                # exit(0)

                # pos[idx, :lengths_x[idx]] = tmp_d
                # pos[idx] = np.append(pos[idx], pos[idx][0])[1:]
                # pos[idx] = np.tanh(pos[idx] / 1000)
                '''
                第一种:pos 正样本为0,副样本为1
                pos[idx, :lengths_x[idx]] = [
                    0. if (float(duration_max_size) / stand > _ > float(duration_size) / stand) else 1. for _ in tmp_d]
                '''
                '''
                pos[idx, :lengths_x[idx]] = tmp_d
                arr=get_pos_w(x[idx])
                pos[idx] = pos[idx] * arr
                pos[idx] = np.append(pos[idx], pos[idx][0])[1:]
                pos[idx] = np.tanh(pos[idx] / 1000)
                '''
            #     arr = get_pos_w(x[idx])
            #     pos[idx] = arr
            #     pos[idx] = np.append(pos[idx], pos[idx][0])[1:]
            # else:
            arr = get_pos_w(x[idx])
            pos[idx] = arr
            pos[idx] = np.append(pos[idx], pos[idx][0])[1:]


    if type(y_origin[0]) is list:
        # train
        y = np.zeros((n_samples, max_len)).astype('int32')
        lengths_y = [len(s) for s in y_origin]
        for idx, sent in enumerate(y_origin):
            y[idx, :lengths_y[idx]] = sent
            # pos[idx, :lengths_y[idx]] = [0. if (_ > 300) else 1. for _ in tmp_d]
            # print('x---{}'.format(x[idx, :lengths_x[idx]]))
            # print('y---{}'.format(y[idx, :lengths_y[idx]]))
            # print('d---{}'.format(d[idx, :int(np.sum(mask[idx]))]))
            # print('d---{}'.format(mask[idx, :lengths_x[idx]]))
            # exit(0)
    else:
        # test
        y = np.array(y_origin).astype('int32')

    # v:将x的物品序列转化为one-hot序列
    # 原来的x:[[1 9 5 ...],[...]]
    # 转化后的x:[[[1 0 0 0 ..],[0 0 0 0 ...]...],[[...],[...]]]
    if one_hot:
        # n_samples:案例数量    max_len:物品序列长度   vocab_size:物品one-hot向量的长度
        one_hot_x = np.zeros((n_samples, max_len, vocab_size)).astype('int32')
        for i in range(n_samples):
            for j in range(max_len):
                one_hot_x[i, j, x[i, j]] = 1
        x = one_hot_x
    else:
        x = x.reshape(x.shape[0], x.shape[1], ndim)

    # mask:[[1 1 1 1 1...0 0 0 0 0],[1 1 1 ... 0 0]] 1的个数表示物品的长度
    # lengths_x:[1519 1596 ...] 每一个数字表示用户的序列长度
    # y:next game id的list [0 0 0 1 0 ...] 0为英雄联盟
    ret = {'x': x, 'y': y, 'mask': mask, 'lengths': lengths_x}
    if t_origin is not None:
        ret['t'] = t
        # addv
        if d_origin is not None:
            ret['d'] = d
    ret['pos'] = pos

    return ret


def save_model(filename, suffix, model, log=None, announce=True, log_only=False):
    # Build filename
    filename = '{}_{}'.format(filename, suffix)
    # Store in separate directory
    filename = os.path.join('./models/', filename)
    # Inform user
    if announce:
        logging.info('Saving to: {}'.format(filename))
    # Generate parameter filename and dump
    param_filename = '%s.params' % (filename)
    if not log_only:
        # Acquire Data
        data = lasagne.layers.get_all_param_values(model)
        with open(param_filename, 'w') as f:
            pickle.dump(data, f)
    # Generate log filename and dump
    if log is not None:
        log_filename = '%s.log' % (filename)
        with open(log_filename, 'w') as f:
            pickle.dump(log, f)


def get_pos_w(arr):
    tmp_arr = []

    def check(x):
        if x in tmp_arr:
            return 0.0
        else:
            tmp_arr.append(x)
            return 1

    check_func = np.frompyfunc(check, 1, 1)
    pos_w = check_func(arr)
    return pos_w
