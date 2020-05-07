# -*- coding: utf-8 -*-
# ! /usr/bin/env python

import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne import nonlinearities
from lasagne import init
from lasagne.utils import unroll_scan

from lasagne.layers.base import MergeLayer, Layer
from lasagne.layers.input import InputLayer
from lasagne.layers.dense import DenseLayer
from lasagne.layers.recurrent import Gate
from lasagne.layers import BatchNormLayer
from tgate import OutGate, TimeGate


class VTLSTM2Layer(MergeLayer):
    def __init__(self,
                 incoming,  # 输入层输出	(batch size, SEQ_LENGTH, num_features)
                 num_units,  # 隐藏层单元个数 (128)
                 time_input,  # 输入层时间 (batch size, SEQ_LENGTH)
                 ingate=Gate(),
                 forgetgate=Gate(),
                 cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 outgate=OutGate(),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 peepholes=True,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,  # 输入层有效序列(1 1 1 1 1 1 ... 0 0 0 0) (batch size, SEQ_LENGTH)
                 only_return_final=False,
                 bn=False,
                 tgate1=TimeGate(W_t=init.Uniform((-1, 0))),  # add 添加时间门
                 tgate2=TimeGate(),  # add2 添加时间门2
                 boundary=-0.00001,  # add2 不知道什么用 constraint ceil
                 **kwargs):
        # 建立incomings作为所有输入层的list,并将incoming作为第一个元素
        incomings = [incoming]
        # add 时间作为必要输入
        incomings.append(time_input)
        self.time_incoming_index = len(incomings) - 1

        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        self.cell_init_incoming_index = -1

        # v:MergeLayer可以有多个输入层,可以使用append将输入层叠加,然后调用父类的__init__初始化
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings) - 1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings) - 1
        if isinstance(cell_init, Layer):
            incomings.append(cell_init)
            self.cell_init_incoming_index = len(incomings) - 1

        # Initialize parent layer
        super(VTLSTM2Layer, self).__init__(incomings, **kwargs)

        # If the provided nonlinearity is None, make it linear
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        # v:多个变量不知道什么意思
        self.learn_init = learn_init  # 不知道什么意思 default:false 可能没有什么用
        self.num_units = num_units  # default:128
        self.backwards = backwards  # 不知道什么意思 default:false
        self.peepholes = peepholes  # 不知道什么意思 default:true
        self.gradient_steps = gradient_steps  # 不知道什么意思 default:-1
        self.grad_clipping = grad_clipping  # 不知道什么意思 default:0
        self.unroll_scan = unroll_scan  # 不知道什么意思 default:false
        self.precompute_input = precompute_input  # 不知道什么意思 default:false
        self.only_return_final = only_return_final  # 不知道什么意思 default:false
        self.boundary = boundary  # add2

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # 验证输入向量
        # input_shapes是自带的方法,用于查看输入的维度
        input_shape = self.input_shapes[0]
        # add
        time_shape = self.input_shapes[1]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        # 返回给定轴上的数组元素的乘积。
        num_inputs = np.prod(input_shape[2:])

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        def add_outgate_params(gate, gate_name):
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.W_to, (1, num_units),
                                   name="W_to_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        # add
        def add_timegate_params(gate, gate_name):
            return (self.add_param(gate.W_t, (1, num_units),
                                   name="W_t_to_{}".format(gate_name)),
                    self.add_param(gate.W_x, (num_inputs, num_units),
                                   name="W_x_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name)),
                    gate.nonlinearity_inside,
                    gate.nonlinearity_outside
                    )

        # 添加LSTM的输入门
        (self.W_in_to_ingate, self.W_hid_to_ingate, self.b_ingate,
         self.nonlinearity_ingate) = add_gate_params(ingate, 'ingate')
        # 添加LSTM的忘记门
        (self.W_in_to_forgetgate, self.W_hid_to_forgetgate, self.b_forgetgate,
         self.nonlinearity_forgetgate) = add_gate_params(forgetgate, 'forgetgate')
        # 添加LSTM的单元(cell)
        (self.W_in_to_cell, self.W_hid_to_cell, self.b_cell,
         self.nonlinearity_cell) = add_gate_params(cell, 'cell')
        # 添加LSTM的输出门
        (self.W_in_to_outgate, self.W_hid_to_outgate, self.W_to_to_outgate,
         self.b_outgate,
         self.nonlinearity_outgate) = add_outgate_params(outgate, 'outgate')
        # add
        (self.W_t1_to_tg1, self.W_x1_to_tg1, self.b1_tg1, self.nonlinearity_inside_tg1,
         self.nonlinearity_outside_tg1) = add_timegate_params(tgate1, 'tgate1')

        # add2
        (self.W_t2_to_tg2, self.W_x2_to_tg2, self.b2_tg2, self.nonlinearity_inside_tg2,
         self.nonlinearity_outside_tg2) = add_timegate_params(tgate2, 'tgate2')

        # 如果启用了peepholes（单元到门）连接，则初始化peepholes连接。
        # 即cell的输出会通到输入门,输出门,忘记门
        if self.peepholes:
            self.W_cell_to_ingate = self.add_param(
                ingate.W_cell, (num_units,), name="W_cell_to_ingate")

            self.W_cell_to_forgetgate = self.add_param(
                forgetgate.W_cell, (num_units,), name="W_cell_to_forgetgate")

            self.W_cell_to_outgate = self.add_param(
                outgate.W_cell, (num_units,), name="W_cell_to_outgate")

        # 这两个单元就是cell和hid,一下第一次初始化
        if isinstance(cell_init, Layer):
            self.cell_init = cell_init
        else:
            self.cell_init = self.add_param(
                cell_init, (1, num_units), name="cell_init",
                trainable=learn_init, regularizable=False)

        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

        # 如果bn为true,则构造BatchNormLayer,This layer implements batch normalization of its inputs.
        # self.params.update(self.bn.params)?似乎是对所有的参数进行标准化
        if bn:
            self.bn = lasagne.layers.BatchNormLayer(input_shape, axes=(0, 1))
            self.params.update(self.bn.params)
        else:
            self.bn = False

    # 输出lstm层的维度(None,None,128)
    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], self.num_units
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, inputs, **kwargs):
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        cell_init = None

        # 取出之前存储在inputs中的几个输入
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]

        time_mat = inputs[self.time_incoming_index]

        # 如果ndim>3,则折叠input的后面的尺寸
        '''
        例如，如果我们用展平（x,outdim = 2）展平形状（2,3,4,5）的张量，
        那么我们将具有相同的（2-1 = 1）前导尺寸（2,）,
        其余尺寸已折叠.因此,此示例中的输出将具有形状（2,60）.
        '''
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # 批量正则化
        if self.bn:
            input = self.bn.get_output_for(input)

        # 交换1 2维的数据
        input = input.dimshuffle(1, 0, 2)
        # (n_time_steps, n_batch)
        # add
        time_input = time_mat.dimshuffle(1, 0, 'x')
        time_seq_len, time_num_batch, _ = time_input.shape
        seq_len, num_batch, _ = input.shape

        # 合成 num_features,(num_units*6)) vector
        # 同时添加一个权重矩阵
        W_in_stacked = T.concatenate(
            [self.W_in_to_ingate,
             self.W_in_to_forgetgate,
             self.W_in_to_cell,
             self.W_in_to_outgate,
             self.W_x2_to_tg2,  # add2
             self.W_x1_to_tg1],  # add
            axis=1)

        W_hid_stacked = T.concatenate(
            [self.W_hid_to_ingate,
             self.W_hid_to_forgetgate,
             self.W_hid_to_cell,
             self.W_hid_to_outgate
             ], axis=1)
        # 合成 (6*num_units) vector
        b_stacked = T.concatenate(
            [self.b_ingate,
             self.b_forgetgate,
             self.b_cell,
             self.b_outgate,
             self.b2_tg2,  # add2
             self.b1_tg1  # add 添加时间的偏置
             ], axis=0)

        # add2 永用于控制Wt1 使用 W_t1_to_tg1_constraint 代替 W_t1_to_tg1
        # W_t1_to_tg1_constraint < 0
        W_t1_to_tg1_constraint = T.switch(T.ge(self.W_t1_to_tg1, self.boundary), self.W_t1_to_tg1, self.boundary)

        # add t只有和两个矩阵相乘
        # Stack delta time weight matrices into a (1, 2* num_units)
        W_t_stacked = T.concatenate([
            self.W_to_to_outgate,
            self.W_t2_to_tg2,  # add
            W_t1_to_tg1_constraint  # change
        ], axis=1)

        if self.precompute_input:
            # Because the input is given for all time steps, we can
            # precompute_input the inputs dot weight matrices before scanning.
            # W_in_stacked is (n_features, 4*num_units). input is then
            # (n_time_steps, n_batch, 4*num_units).
            # add 输入预计算
            time_input = T.dot(time_input, W_t_stacked)
            input = T.dot(input, W_in_stacked) + b_stacked

        # When theano.scan calls step, input_n will be (n_batch, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        # change
        def slice_w(x, start, stride=1):
            return x[:, start * self.num_units:(start + stride) * self.num_units]

        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step(input_n, time_input_n, cell_previous, hid_previous, *args):
            # 之前已经有预计算的时候,这里不用,但是不知道什么用
            # 可能在这里是分步计算
            if not self.precompute_input:
                # add
                # time_input_n是time序列中的一个输入
                # 之前time_input_n(n_batch,'x')
                # time_input_n(n_time_steps, n_batch,'x')
                time_input_n = T.dot(time_input_n, W_t_stacked)
                # 之前input_n(n_batch, n_features)
                # input_n(n_time_steps, n_batch, num_units)
                input_n = T.dot(input_n, W_in_stacked) + b_stacked

            # 通过分片的函数,将输入的数据分成几个部分,
            # 这几个部分分别对应一块,
            # 如 tm_wto_n表示t和wto相乘的块
            # add
            tm_wto_n = slice_w(time_input_n, 0)

            # add2
            tm_w2_n = slice_w(time_input_n, 1)
            tm_w1_n = slice_w(time_input_n, 2)
            tm_w2_n = self.nonlinearity_inside_tg2(tm_w2_n)
            tm_w1_n = self.nonlinearity_inside_tg1(tm_w1_n)
            tm2_xwb_n = slice_w(input_n, 4)
            tm1_xwb_n = slice_w(input_n, 5)
            timegate2 = self.nonlinearity_outside_tg2(tm_w2_n + tm2_xwb_n)
            timegate1 = self.nonlinearity_outside_tg1(tm_w1_n + tm1_xwb_n)

            input_n = slice_w(input_n, 0, 4)

            # Calculate gates pre-activations and slice
            gates = input_n + T.dot(hid_previous, W_hid_stacked)

            # Clip gradients
            if self.grad_clipping:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)

            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)
            # add outgate 的在输出到激活函数之前,会添加一个tm
            outgate += tm_wto_n

            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous * self.W_cell_to_ingate
                forgetgate += cell_previous * self.W_cell_to_forgetgate

            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)

            # Compute new cell value
            # add 在cell中,后面的乘项增加一个
            # cell = forgetgate * cell_previous + ingate * timegate1 * cell_input
            # add2
            cell = forgetgate * cell_previous + ingate * timegate2 * cell_input
            tilde_cell = forgetgate * cell_previous + ingate * timegate1 * cell_input

            if self.peepholes:
                outgate += cell * self.W_cell_to_outgate
            outgate = self.nonlinearity_outgate(outgate)

            # Compute new hidden unit activation
            # hid = outgate * self.nonlinearity(cell)
            # add2 change
            hid = outgate * self.nonlinearity(tilde_cell)
            return [cell, hid]

        def step_masked(input_n,
                        time_input_n,  # add 添加时间的输入
                        mask_n, cell_previous, hid_previous, *args):
            cell, hid = step(input_n,
                             time_input_n,  # add 添加时间的输入
                             cell_previous, hid_previous, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)

            return [cell, hid]

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            # (1, 0, ‘x’) -> AxB to BxAx(可广播的维度)
            mask = mask.dimshuffle(1, 0, 'x')
            # input(seq_len,batch_size,n_feature),mask(seq_len, batch_size,(可广播的维度))
            # add 这里设置sequences,可能是用于后面的scan
            sequences = [input, time_input, mask]
            step_fun = step_masked
        else:
            # add
            sequences = [input, time_input]
            step_fun = step

        # 后面不太懂
        ones = T.ones((num_batch, 1))
        if not isinstance(self.cell_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            # ones(num_batch,1) self.cell(1,num_units)
            cell_init = T.dot(ones, self.cell_init)

        if not isinstance(self.hid_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            # ones(num_batch,1) self.hid(1,num_units)
            hid_init = T.dot(ones, self.hid_init)

        # The hidden-to-hidden weight matrix is always used in step
        # 权重属于不变的量
        non_seqs = [W_hid_stacked]
        # The "peephole" weight matrices are only used when self.peepholes=True
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate,
                         self.W_cell_to_outgate]

        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        # 之前没有计算的话,就需要
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            cell_out, hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            cell_out, hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,  # [input, time_input, mask]
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                truncate_gradient=self.gradient_steps,
                non_sequences=non_seqs,
                strict=True)[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        return hid_out
