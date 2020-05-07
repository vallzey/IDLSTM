# -*- coding: utf-8 -*-
# ! /usr/bin/env python

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

import vutils, vutils_tr, vutils_te

# addv 修改模型
from vtlstm2 import VTLSTM2Layer
from vdtlstm_22_modify_06 import VDTLSTMLayer
from vdtlstm_22_modify_07 import VDTLSTMEMLayer

from tgate import OutGate, TimeGate
import ConfigParser
import pickle

theano.config.floatX = 'float32'
theano.config.on_unused_input = 'ignore'

parser = argparse.ArgumentParser(description='Specific model, data and other params.')
parser.add_argument('--model', type=str, default='DTLSTM',
                    help='Model to train:LSTM, LSTM_T, PLSTM, TLSTM1, TLSTM2, TLSTM3,DTLSTM.')
parser.add_argument('--data', type=str, default='music', help='Input data source: music, citeulike, game.')
parser.add_argument('--fixed_epochs', type=int, default=10, help='Number of epochs in the first stage.')
parser.add_argument('--num_epochs', type=int, default=20,
                    help='Number of epochs in the first and second stage.')  # 循环次数
parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden unit.')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--sample_time', type=int, default=3,
                    help='Sample time in the evaluate method.')  # evaluate方法中的采样次数

parser.add_argument('--batch_size', type=int, default=5, help='Batch size in the training phase.')
parser.add_argument('--test_batch', type=int, default=5, help='Batch size in the testing phase')
# 需要修改
parser.add_argument('--vocab_size', type=int, default=5000, help='Vocabulary size 500 1050')  # 当one-hot时,一般指物品数量
# 需要修改
parser.add_argument('--duration_size', type=float, default=500, help='Vocabulary size 500 1050')  # 当one-hot时,一般指物品数量

parser.add_argument('--duration_max_size', type=float, default=10000, help='duration_size')  #
parser.add_argument('--stand', type=float, default=1, help='standardization')  #

# 需要修改
parser.add_argument('--max_len', type=int, default=200, help='Maximum length of the sequence.')  # 指序列长度
parser.add_argument('--grad_clip', type=int, default=0,
                    help='Maximum grad step. Grad will be cliped if greater than this. 0 means no clip')
parser.add_argument('--debug', dest='debug', action='store_true',
                    help='If debug is set, train one time, load small dataset.')
parser.add_argument('--bn', dest='bn', action='store_true', help='If bn is set, input data will be batch normed')
parser.add_argument('--sigmoid_on', dest='sigmoid_on', action='store_true',
                    help='if sigmoid_on is set, input time data will be sigmoid')
parser.add_argument('--rank', type=int, default=10, help='recall@rank.')
parser.set_defaults(debug=False)
parser.set_defaults(sigmoid_on=False)
parser.set_defaults(bn=False)
args = parser.parse_args()

#######################################################
# Assign the args values to global variables

DEBUG = args.debug
SIGMOID_ON = args.sigmoid_on
# batch norm
BN = args.bn
# Data source
DATA_TYPE = args.data  # citeulike, music
DATA_TYPE = DATA_TYPE.replace('\r', '')
# Sequence Length
SEQ_LENGTH = args.max_len
# Vocabulary size
VOCAB_SIZE = args.vocab_size

DURATION_SIZE = args.duration_size
DURATION_MAX_SIZE = args.duration_max_size
STAND = args.stand

# LSTM_T, PLSTM, TLSTM, TLSTM1, TLSTM2, TLSTM3
MODEL_TYPE = args.model
# Hidden unit
# v:隐藏层单元个数
N_HIDDEN = args.num_hidden
# Optimization learning rate
LEARNING_RATE = args.learning_rate
# All gradients above this will be clipped
GRAD_CLIP = args.grad_clip
# Number of epochs to train the net

NUM_EPOCHS = args.num_epochs
# Number of epochs in the first phase
FIXED_EPOCHS = args.fixed_epochs
# Batch Size
BATCH_SIZE = args.batch_size
TEST_BATCH = args.test_batch
# Number of units in the two hidden (LSTM) layers
SAMPLE_TIME = args.sample_time
# 打印频次
PRINT_FREQ = 20
# Use one hot vector to represent input data
ONE_HOT = True
if DEBUG:
    PRINT_FREQ = 1
#######################################################
# Set data load format
# input layer contains Time if True
USE_TIME_INPUT = False
NDIM = 1 if not ONE_HOT else VOCAB_SIZE
# USE_TIME_INFO and USE_DELTA_TIME decite load data format
USE_TIME_INFO = False
USE_DELTA_TIME = False

RANK = args.rank

# addv
USE_DURATION = False
if MODEL_TYPE in ['DTLSTM', 'DTLSTM_EM']:
    USE_TIME_INPUT = True
    USE_DELTA_TIME = True
    USE_DURATION = True

elif MODEL_TYPE in ['TLSTM2']:
    USE_TIME_INPUT = True
    USE_DELTA_TIME = True
elif MODEL_TYPE == 'LSTM_T':
    USE_TIME_INPUT = True
    USE_DELTA_TIME = True
elif MODEL_TYPE == 'LSTM':
    pass
else:
    print("Wrong Modle specified {}".format(MODEL_TYPE))
    exit()

# Set random seed for lasagne
lasagne.random.set_rng(np.random.RandomState(1))

if not os.path.exists('log'):
    os.makedirs('log')

# Initial logger
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(funcName)s:%(lineno)s] %(message)s"
if DEBUG:
    logging.basicConfig(filename='log/DEBUG-{}-{}-{}.log'.format(MODEL_TYPE, DATA_TYPE, str(datetime.datetime.now())),
                        level=logging.INFO, format=FORMAT)
else:
    logging.basicConfig(
        filename='log/{}-{}-{}-{}.log'.format(MODEL_TYPE, DATA_TYPE, N_HIDDEN, str(datetime.datetime.now())),
        level=logging.INFO, format=FORMAT)

# v:设置控制太输出的handler
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(FORMAT))
logging.getLogger().addHandler(handler)

logging.info('Start {} {}'.format(MODEL_TYPE, DATA_TYPE))
logging.info('VOCAB_SIZE {}, MAX_LEN {}, HIDDEN {}'.format(VOCAB_SIZE, SEQ_LENGTH, N_HIDDEN))
for k, v in locals().items():
    logging.info('{}  {}'.format(k, v))

# Load train data, test data to a dictionary
DATA_ATTR = {
    'max_len': SEQ_LENGTH,
    'vocab_size': VOCAB_SIZE,
    'debug': DEBUG,
    'source': DATA_TYPE,
    'with_time': USE_TIME_INFO,
    'with_delta_time': USE_DELTA_TIME,
    'with_duration': USE_DURATION,  # addv
    'duration_size': DURATION_SIZE,
    'duration_max_size': DURATION_MAX_SIZE,
    'stand': STAND,

}
logging.info('Data {}'.format(DATA_ATTR))

# addc
train_data = vutils_tr.load_data(DATA_ATTR)
test_data = vutils_te.load_data(DATA_ATTR)

train_data_size = len(train_data['x'])
test_data_size = len(test_data['x'])


# 根据p(当前位置),按照batch_size获取data
# 后面在训练的时候会再次用到,这里先用于处理test集合
def gen_data(p, data, batch_size=1):
    # generate data for the model
    # y in train data is a matrix (batch_size, seq_length)
    # y in test data is an array
    x = data['x'][p:p + batch_size]
    y = data['y'][p:p + batch_size]
    batch_data = {'x': x, 'y': y}
    if data.has_key('t'):
        batch_data['t'] = data['t'][p:p + batch_size]
        # addv
        if data.has_key('d'):
            batch_data['d'] = data['d'][p:p + batch_size]

    ret = vutils.prepare_data(batch_data, VOCAB_SIZE, one_hot=ONE_HOT, sigmoid_on=SIGMOID_ON,
                              duration_size=DURATION_SIZE, duration_max_size=DURATION_MAX_SIZE, stand=STAND)
    return ret


# 为什么取batch_size=len(test_data['x'])
test_data = gen_data(0, test_data, batch_size=len(test_data['x']))
# v:此时x已经是三维的one-hot向量 这里去x的第二个维度,即序列长度最长值
test_data_length = test_data['x'].shape[1]

logging.info("Test x shape {}".format(test_data['x'].shape))
# 此时test_data['x']是numpy类型
# 而train_data['x']是list类型 不知道为什么?
logging.info("Train x length {}".format(len(train_data['x'])))


def main(num_epochs=NUM_EPOCHS, vocab_size=VOCAB_SIZE):
    logging.info("Building network ...")
    # (batch size, SEQ_LENGTH, num_features)
    # v: None表示该维度的大小在编译时没有固定。
    # InputLayer，它可用于表示网络的输入。张量的第一个维度通常是批量维度
    l_in = lasagne.layers.InputLayer(shape=(None, None, NDIM))

    # logging.info("读取embedding模型 ...")
    # w = pickle.load(open('G_1050', 'rb')).astype(np.float32)
    # (i, o) = w.shape
    #
    # l_in2 = lasagne.layers.DenseLayer(l_in, num_units=o, W=lasagne.init.Normal(),
    #                                   num_leading_axes=2, nonlinearity=None)

    units = 128
    logging.info("Embedding num_units : {}".format(units))

    l_in2 = lasagne.layers.DenseLayer(l_in, num_units=units, W=lasagne.init.Normal(),
                                      num_leading_axes=2, nonlinearity=None)

    # l_in2 = lasagne.layers.EmbeddingLayer(l_in, input_size=i, output_size=o, W=w)

    l_mask = lasagne.layers.InputLayer(shape=(None, None))

    # addv
    l_pos = lasagne.layers.InputLayer(shape=(None, None))

    # We now build the LSTM layer which takes l_in as the input layer
    # We clip the gradients at GRAD_CLIP to prevent the problem of exploding gradients.
    l_forward = None

    if MODEL_TYPE == 'DTLSTM':
        l_t = lasagne.layers.InputLayer(shape=(None, None))
        l_d = lasagne.layers.InputLayer(shape=(None, None))
        l_forward = VDTLSTMLayer(
            l_in2,
            time_input=l_t,
            duration_input=l_d,
            num_units=N_HIDDEN,
            mask_input=l_mask,
            peepholes=True,
            ingate=lasagne.layers.Gate(),
            forgetgate=lasagne.layers.Gate(),
            cell=lasagne.layers.Gate(W_cell=None, nonlinearity=lasagne.nonlinearities.tanh),
            outgate=OutGate(),
            nonlinearity=lasagne.nonlinearities.tanh,
            cell_init=lasagne.init.Constant(0.),
            hid_init=lasagne.init.Constant(0.),
            grad_clipping=GRAD_CLIP,
            only_return_final=False,
            bn=BN,
        )
    elif MODEL_TYPE == 'DTLSTM_EM':
        l_t = lasagne.layers.InputLayer(shape=(None, None))
        l_d = lasagne.layers.InputLayer(shape=(None, None))
        l_forward = VDTLSTMEMLayer(
            l_in2,
            time_input=l_t,
            duration_input=l_d,
            num_units=N_HIDDEN,
            mask_input=l_mask,
            peepholes=True,
            ingate=lasagne.layers.Gate(),
            cell=lasagne.layers.Gate(W_cell=None, nonlinearity=lasagne.nonlinearities.tanh),
            outgate=OutGate(),
            nonlinearity=lasagne.nonlinearities.tanh,
            cell_init=lasagne.init.Constant(0.),
            hid_init=lasagne.init.Constant(0.),
            grad_clipping=GRAD_CLIP,
            only_return_final=False,
            bn=BN,
        )
    elif MODEL_TYPE == 'TLSTM2':
        l_t = lasagne.layers.InputLayer(shape=(None, None))
        l_forward = VTLSTM2Layer(
            l_in2,
            time_input=l_t,
            num_units=N_HIDDEN,
            mask_input=l_mask,
            peepholes=True,
            ingate=lasagne.layers.Gate(),
            forgetgate=lasagne.layers.Gate(),
            cell=lasagne.layers.Gate(W_cell=None, nonlinearity=lasagne.nonlinearities.tanh),
            outgate=OutGate(),
            nonlinearity=lasagne.nonlinearities.tanh,
            cell_init=lasagne.init.Constant(0.),
            hid_init=lasagne.init.Constant(0.),
            grad_clipping=GRAD_CLIP,
            only_return_final=False,
            bn=BN,
        )
    else:
        logging.info('没有这种模型类型')
        exit(0)

    target_values = T.matrix('target_values', dtype='int32')

    # v:输出层(N_HIDDEN,vocab_size)
    # 调用了l_forward中get_output_shape_for()方法
    # l_forward (num_batch, sequence_length, num_units)
    l_out = lasagne.layers.DenseLayer(l_forward, num_units=vocab_size, W=lasagne.init.Normal(),
                                      num_leading_axes=2, nonlinearity=None)

    # 获取输出层的输出(None, None, 500)
    # 调用了l_forward中get_output_for()方法
    # l_out (num_batch, sequence_length, vocab_size)
    network_output = lasagne.layers.get_output(l_out)

    # (2, 0, 1) -> AxBxC to CxAxB
    # (0, ‘x’, 1) -> AxB to Ax1xB
    # (1, ‘x’, 0) -> AxB to Bx1xA
    # (sequence_length, num_batch, vocab_size)
    network_output = network_output.dimshuffle(1, 0, 2)

    def calculate_softmax(n_input):
        return T.nnet.softmax(n_input)

    def merge_cost(n_input, n_target, n_mask, n_pos, cost_prev):
        # 使用ravel将原始矩阵张开
        n_target = n_target.ravel()
        # addv
        # n_pos = T.reshape(n_pos, (5, 1))
        # n_input = n_pos - n_input
        # n_pos = (n_pos - 0.5) * 2
        # n_input = n_input * n_pos

        n_cost = T.nnet.categorical_crossentropy(n_input, n_target)
        n_cost = n_cost * n_mask * n_pos  # * (1.0 - n_pos)
        n_cost = n_cost.sum()
        return cost_prev + n_cost

    network_output_softmax, _ = theano.scan(fn=calculate_softmax, sequences=network_output)

    # The loss function is calculated as the mean of the (categorical) cross-entropy between the prediction and target.
    # 后面用于计算交叉熵损失函数的sum
    m_cost, _ = theano.scan(fn=merge_cost,
                            sequences=[network_output_softmax, target_values.T, l_mask.input_var.T, l_pos.input_var.T],

                            outputs_info=T.constant(0.))
    # m_cost是一个序列,但是只需要最后一个叠加值cost[-1]
    m_cost = m_cost[-1]
    # 求平均cost
    cost = m_cost / l_mask.input_var.sum()

    # 转换回来: (batch_size, time_seqsence, vocab_size)
    network_output_softmax = network_output_softmax.dimshuffle(1, 0, 2)

    # Compute AdaGrad updates for training
    logging.info("Computing updates ...")
    # 这个get_all_params方法应该是用于获取所有的在lstmlayer中add_param
    all_params = lasagne.layers.get_all_params(l_out, trainable=True)
    # 根据cost更新所有的参数all_params,学习率为LEARNING_RATE
    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)

    # Theano functions for training, predict
    logging.info("Compiling functions ...")
    input_var = [l_in.input_var, l_mask.input_var]
    # add
    if USE_TIME_INPUT:
        input_var += [l_t.input_var]
        # addv
        if USE_DURATION:
            input_var += [l_d.input_var]

    predict = theano.function(input_var, network_output_softmax, allow_input_downcast=True)
    input_var += [target_values]
    # addv
    input_var.insert(2, l_pos.input_var)

    # v:计算损失函数值
    # input_var[l_in.input_var, l_mask.input_var, l_pos.input_var,l_t.input_var,l_d.input_var,target_values]
    train = theano.function(input_var, cost, updates=updates, allow_input_downcast=True)
    # compute_cost return cost but without update
    compute_cost = theano.function(input_var, cost, allow_input_downcast=True)

    # v:评估方法!!!!
    # addv
    def do_evaluate(test_x, test_y, test_mask, lengths, test_t=None, test_d=None, n=1000, test_batch=5):
        # evaluate and calculate recall@10, MRR@10
        p = 0
        probs_all_time = None  # 所有的预测值
        while True:
            input_var = [test_x[p:p + test_batch], test_mask[p:p + test_batch]]
            if test_t is not None:
                input_var += [test_t[p:p + test_batch]]
                # addv
                if test_d is not None:
                    input_var += [test_d[p:p + test_batch]]
            batch_probs = predict(*input_var)
            p += test_batch
            probs_all_time = batch_probs if probs_all_time is None else np.concatenate([probs_all_time, batch_probs],
                                                                                       axis=0)
            if p >= test_x.shape[0]:
                break

        total_size = test_x.shape[0]
        recall10 = 0.
        MRR10_score = 0.
        rate_sum = 0

        sample_time = SAMPLE_TIME

        # addv
        _rank = []

        for idx in range(total_size):
            gnd = test_y[idx]
            probs = probs_all_time[idx, lengths[idx] - 1, :]  # 取每一个test的最后一个的预测值,一个500维的向量
            prob_index = np.argsort(probs)[-1::-1].tolist()  # argsort函数返回的是数组值从小到大的索引值[3 1 2]-->[1 2 0]
            gnd_rate = prob_index.index(gnd) + 1
            # 这个是所有的东西的排名
            rate_sum += gnd_rate
            # Sample multiple times to reduce randomness
            for _ in range(sample_time):

                # addvv
                samples = np.random.choice(range(vocab_size), vocab_size, replace=False).tolist()
                # make sure the fist element is gnd
                # v 这样在随机之后,只要选择index(0)知道是第几了
                try:
                    samples.remove(gnd)
                    samples.insert(0, gnd)
                except ValueError:
                    samples[0] = gnd

                sample_probs = probs[samples]
                prob_index = np.argsort(sample_probs)[-1::-1].tolist()
                # v 这个是随机100个的排名
                rate = prob_index.index(0) + 1

                # addvv
                logging.info('rank:{}'.format(rate))

                # caculate Recall@10 and MRR@10
                # addvc
                if rate <= RANK:
                    recall10 += 1
                    MRR10_score += 1. / rate

        count = total_size * sample_time
        recall10 = recall10 / count
        MRR10_score = MRR10_score / count
        avg_rate = float(rate_sum) / total_size

        logging.info('Recall@10 {}'.format(recall10))
        logging.info('MRR@10 1/rate {}'.format(MRR10_score))
        logging.info('Average rate {}'.format(avg_rate))

    def onehot2int(onehot_vec):
        # convert onehot vector to index
        ret = []
        for onehot in onehot_vec:
            ret.append(onehot.tolist().index(1))
        return ret

    def get_short_test_data(length):
        # generate short sequence in the test_data.
        test_x = test_data['x'][:, :length]
        test_mask = test_data['mask'][:, :length]
        # add
        test_t = test_data['t'][:, :length] if USE_TIME_INPUT else None
        # addv
        test_d = test_data['d'][:, :length] if USE_DURATION else None

        lengths = np.sum(test_mask, axis=1).astype('int')

        test_y = test_data['y'].copy()
        for idx in range(test_y.shape[0]):
            whole_length = test_data['lengths'][idx]
            if length < whole_length:
                test_y[idx] = test_data['x'][idx, length, :].tolist().index(1) if ONE_HOT else test_data['x'][
                    idx, length, 0]

        return test_x, test_y, test_mask, lengths, test_t, test_d

    def evaluate(model, current_epoch, additional_test_length):
        # Evaluate the model
        logging.info('Evaluate')
        # 包括了所有测试集合
        test_x = test_data['x']
        test_y = test_data['y']
        test_mask = test_data['mask']
        lengths = test_data['lengths']
        logging.info('-----------Evaluate Normal:{},{},{}-------------------'.format(MODEL_TYPE, DATA_TYPE, N_HIDDEN))
        do_evaluate(test_x, test_y, test_mask, lengths,
                    test_data['t'] if USE_TIME_INPUT else None,
                    test_data['d'] if USE_DURATION else None,
                    test_batch=TEST_BATCH)
        # Evaluate the model on short data
        if additional_test_length > 0:
            logging.info('-----------Evaluate Additional---------------')
            # addv
            test_x, test_y, test_mask, lengths, test_t, test_d = get_short_test_data(additional_test_length)
            do_evaluate(test_x, test_y, test_mask, lengths, test_t, test_d, test_batch=TEST_BATCH)
        logging.info('-----------Evaluate End----------------------')
        if not DEBUG:
            vutils.save_model('{}-{}-{}-{}'.format(MODEL_TYPE, current_epoch, DATA_TYPE, N_HIDDEN),
                              str(datetime.datetime.now()), model, '_new')

    def add_test_to_train(length):
        logging.info('Length {} test cases added to train set'.format(length))
        global train_data
        logging.info('Old train data size {}'.format(len(train_data['x'])))
        # Remote the train_data added before
        train_data['x'] = train_data['x'][:train_data_size]
        train_data['y'] = train_data['y'][:train_data_size]
        if train_data.has_key('t'):
            train_data['t'] = train_data['t'][:train_data_size]
            # addv
            if train_data.has_key('d'):
                train_data['d'] = train_data['d'][:train_data_size]

        test_x = test_data['x']
        lengths = test_data['lengths']
        for idx in range(test_x.shape[0]):
            n_length = length
            # To make sure the complete test case will not be added into train set
            if lengths[idx] <= length:
                n_length = length - 1
            if ONE_HOT:
                # if ONE_HOT is used, we convert one hot vector to int first.
                new_x = onehot2int(test_x[idx, :n_length, :])
                new_y = onehot2int(test_x[idx, 1:n_length + 1, :])
            else:
                new_x = test_x[idx, :n_length, 0]
                new_y = test_x[idx, 1:n_length + 1, 0]
            train_data['x'].append(new_x)
            train_data['y'].append(new_y)
            if train_data.has_key('t'):
                test_t = test_data['t']
                new_t = test_t[idx, :n_length].tolist()
                train_data['t'].append(new_t)

                # addv
                if train_data.has_key('d'):
                    test_d = test_data['d']
                    new_d = test_d[idx, :n_length].tolist()
                    train_data['d'].append(new_d)

        logging.info('New train data size {}'.format(len(train_data['x'])))
        logging.info('--Data Added--')

    logging.info("Training ...")
    logging.info('Data size {},Max epoch {},Batch {}'.format(train_data_size, num_epochs, BATCH_SIZE))
    p = 0
    current_epoch = 0
    it = 0
    data_size = train_data_size
    last_it = 0  # 最后一次迭代的次数
    avg_cost = 0  # 平均损失函数值
    avg_seq_len = 0  # 平均序列长度

    # 随机模块
    plist = vutils.genPlist(data_size, BATCH_SIZE)

    try:
        while True:
            randP = plist[p / BATCH_SIZE]
            batch_data = gen_data(randP, train_data, batch_size=BATCH_SIZE)
            # mask:[[1 1 1 1 1...0 0 0 0 0],[1 1 1 ... 0 0]] 1的个数表示物品的长度
            # lengths_x:[1519 1596 ...] 每一个数字表示用户的序列长度
            # y:next game id的list [0 0 0 1 0 ...] 0为英雄联盟
            x = batch_data['x']
            y = batch_data['y']
            mask = batch_data['mask']
            pos = batch_data['pos']
            avg_seq_len += x.shape[1]

            input_var = [x, mask, pos, y]

            # add
            if USE_TIME_INPUT:
                t = batch_data['t']
                # 消耗时间
                input_var.insert(3, t)
                # addv
                if USE_DURATION:
                    d = batch_data['d']
                    input_var.insert(4, d)
            # v:训练主要方法
            # input_var[x, mask, pos, t, d, y]
            avg_cost += train(*input_var)
            it += 1
            # input_var = [x, mask, t, y]
            p += BATCH_SIZE
            if (p >= data_size):  # 如果p>=data_size,说明一次循环结束
                p = 0
                last_it = it
                current_epoch += 1
                # First stage: Using original train data to train model in #FIXED_EPOCHS
                # Second stage: After that add part of test data to train data.
                # The first stage is using user information with similar interest, and the second stage is using history information
                '''v
                第一阶段：使用原始列车数据在#FIXED_EPOCHS中训练模型
                第二阶段：之后添加部分测试数据来训练数据。
                第一阶段是使用具有类似兴趣的用户信息，第二阶段是使用历史信息.
                '''
                additional_length = int((current_epoch - FIXED_EPOCHS) * test_data_length / (NUM_EPOCHS - FIXED_EPOCHS))
                evaluate(l_out, current_epoch=current_epoch, additional_test_length=additional_length)
                if current_epoch >= num_epochs:
                    break
                if current_epoch > FIXED_EPOCHS:
                    data_size = train_data_size + test_data_size
                    logging.info('>> length {} test cases added to train set.'.format(additional_length))
                    add_test_to_train(additional_length)
                logging.info('Epoch {} Carriage Return'.format(current_epoch))
            if it % PRINT_FREQ == 0:
                # 所以每 PRINT_FREQ * BATCH_SIZE 打印一次
                # current_epoch 循环次数
                logging.info("Epoch {}-{},iter {} average seq length = {} average loss = {}".format(current_epoch, (
                        it - last_it) * 1.0 * BATCH_SIZE / data_size, it, avg_seq_len / PRINT_FREQ,
                                                                                                    avg_cost / PRINT_FREQ))
                avg_cost = 0
                avg_seq_len = 0
        logging.info('End')
    except KeyboardInterrupt:
        logging.info('由于你的自行中断,程序已经停止.')


if __name__ == '__main__':
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('log'):
        os.makedirs('log')
    main(NUM_EPOCHS)
    logging.info('Logging End')
