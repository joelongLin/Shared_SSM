#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@project: PyCharm
@file: testa.py
@author: Shengqiang Zhang
@time: 2020/3/9 16:50
@mail: sqzhang77@gmail.com
"""

from distutils.version import LooseVersion
import tensorflow as tf
from tensorflow.python.layers.core import Dense
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)


import numpy as np
#np.set_printoptions(threshold=np.inf)
import time
import json



def main(_):


    time_feature_lstm = []
    for k in range(3):
        cell = tf.nn.rnn_cell.LSTMCell(num_units=10)
        cell = tf.nn.rnn_cell.ResidualWrapper(cell) if k > 0 else cell
        cell = (
            tf.nn.rnn_cell.DropoutWrapper(cell, state_keep_prob=1.0 - 0.5)
            if 0.5 > 0.0
            else cell
        )
        time_feature_lstm.append(cell)



    # 输入tensors
    # 你看看這個
    # 我的 placeholder 好像每一维都是已知的
    X = tf.placeholder(tf.float32, [20, 10, 10])

    num_units = [100, 200, 300]

    cells = tf.nn.rnn_cell.MultiRNNCell(time_feature_lstm)
    time_rnn_out, _ = tf.nn.dynamic_rnn(
        cell=cells,
        inputs=X,
        initial_state=None,
        dtype=tf.float32,
    )




    configuration = tf.compat.v1.ConfigProto()
    configuration.gpu_options.allow_growth = True


    with tf.compat.v1.Session(config=configuration) as sess:
        sess.run(tf.global_variables_initializer())


if __name__ == '__main__':
   tf.app.run()

