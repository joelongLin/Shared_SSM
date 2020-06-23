# -*- coding: UTF-8 -*-
# author : joelonglin
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import pandas as pd
import pickle
import time
import os
from tqdm import tqdm
import logging
import matplotlib

from ..utils import get_model_params_name

matplotlib.use('Agg') # 用这句话避免 xshell 调参时，出现X11转发
from gluonts.time_feature import time_features_from_frequency_str

from gluonts.model.forecast import SampleForecast
from gluonts.transform import (Chain ,
                               AddInitTimeFeature,
                               AddObservedValuesIndicator,
                               AddTimeFeatures,
                               SwapAxes,
                               InstanceSplitter)
from gluonts.transform.sampler import TestSplitSampler
from gluonts.dataset.field_names import FieldName
from .scaler import MeanScaler , NOPScaler
from .filter import MultiKalmanFilter
from ..utils import create_dataset_if_not_exist , complete_batch , make_nd_diag
from .data_loader import TrainDataLoader_OnlyPast ,InferenceDataLoader_WithFuture, mergeIterOut , stackIterOut
from ..utils import (del_previous_model_params
                    ,time_format_from_frequency_str
                    , plot_train_epoch_loss
                    , plot_train_pred
                    , samples_with_mean_cov
                    , add_time_mark_to_file
                    , add_time_mark_to_dir)

#一些公用字符串
INIT_TIME = 'init_time'


class SharedSSM(object):
    def __init__(self, config , sess):
        self.config = config
        self.sess = sess


        #导入原始的数据 , 确定了 SSM 的数量 ， env_dim 的维度，将ListData对象存放在list中
        self.load_original_data()
        #将导入的原始数据放入dataloader中, 使其能产生placeholders需要的内容
        self.transform_data()
        # 开始搭建有可能Input 对应的 placeholder
        self.placeholders = {
            "past_environment": tf.placeholder(dtype=tf.float32,shape=[config.batch_size, config.past_length, self.env_dim],name="past_environment"),
            "past_env_observed": tf.placeholder(dtype=tf.float32,shape=[config.batch_size, config.past_length, self.env_dim],name="past_env_observed"),
            "pred_environment" : tf.placeholder(dtype=tf.float32, shape=[config.batch_size ,config.pred_length,self.env_dim], name="pred_environment"),
            "pred_env_observed" : tf.placeholder(dtype=tf.float32, shape=[config.batch_size ,config.pred_length,self.env_dim], name="pred_env_observed"),

            "env_past_time_feature": tf.placeholder(dtype=tf.float32,shape=[config.batch_size, config.past_length, self.time_dim],name="env_past_time_feature"),
            "env_pred_time_feature": tf.placeholder(dtype=tf.float32,shape=[config.batch_size, config.pred_length, self.time_dim],name="env_pred_time_feature"),
            "target_past_time_feature": tf.placeholder(dtype=tf.float32,shape=[ config.batch_size, config.past_length, self.time_dim],name="target_past_time_feature"),
            "target_pred_time_feature": tf.placeholder(dtype=tf.float32,shape=[ config.batch_size, config.pred_length, self.time_dim],name="target_pred_time_feature"),

            "past_target" : tf.placeholder(dtype=tf.float32, shape=[self.ssm_num ,config.batch_size , config.past_length, 1], name="past_target"),
            "past_target_observed" : tf.placeholder(dtype=tf.float32, shape=[self.ssm_num ,config.batch_size , config.past_length, 1], name="past_target_observed"),
            "pred_target" : tf.placeholder(dtype=tf.float32, shape=[self.ssm_num ,config.batch_size ,config.pred_length, 1], name="pred_target"),
            "pred_target_observed" : tf.placeholder(dtype=tf.float32, shape=[self.ssm_num ,config.batch_size ,config.pred_length, 1], name="pred_target_observed")
        }


    def load_original_data(self):
        name_prefix = 'data_process/processed_data/{}_{}_{}_{}.pkl'
        # 导入 target 以及 environment 的数据
        if self.config.slice == 'overlap':
            series = self.config.timestep - self.config.past_length - self.config.pred_length + 1
            print('每个数据集的序列数量为 ,', series)
        elif self.config.slice == 'nolap':
            series = self.config.timestep // (self.config.past_length + self.config.pred_length)
            print('每个数据集的序列数量为 ,', series)
        else:
            series = 1
            print('当前序列数量为',series ,',是单长序列的情形')
        # 目标序列的数据路径
        target_path = {name:name_prefix.format(
            '%s_start(%s)_freq(%s)'%(name, self.config.start ,self.config.freq), '%s_DsSeries_%d' % (self.config.slice, series),
            'train_%d' % self.config.past_length, 'pred_%d' % self.config.pred_length,
        ) for name in self.config.target.split(',')}
        target_start = pd.Timestamp(self.config.start,freq=self.config.freq)
        env_start = target_start - self.config.maxlags * target_start.freq
        env_start = env_start.strftime(time_format_from_frequency_str(self.config.freq))
        print('environment 开始的时间：', env_start)
        env_path = {name : name_prefix.format(
            '%s_start(%s)_freq(%s)'%(name, env_start , self.config.freq), '%s_DsSeries_%d' % (self.config.slice, series),
            'train_%d' % self.config.past_length, 'pred_%d' % self.config.pred_length,
        ) for name in self.config.environment.split(',')}

        create_dataset_if_not_exist(
            paths=target_path,start=self.config.start ,past_length=self.config.past_length
            , pred_length=self.config.pred_length,slice=self.config.slice
            , timestep=self.config.timestep, freq=self.config.freq
        )

        create_dataset_if_not_exist(
            paths=env_path, start=env_start, past_length=self.config.past_length
            , pred_length=self.config.pred_length, slice=self.config.slice
            , timestep=self.config.timestep, freq=self.config.freq
        )

        self.target_data , self.env_data = [] , []
        # 由于 target 应该都是 dim = 1 只是确定有多少个 SSM 而已
        for target_name in target_path:
            target = target_path[target_name]
            with open(target, 'rb') as fp:
                target_ds = pickle.load(fp)
                assert target_ds.metadata['dim'] == 1 , 'target 序列的维度都应该为1'
                self.target_data.append(target_ds)

        self.ssm_num = len(self.target_data)
        self.env_dim = 0
        for env_name in env_path:
            env =  env_path[env_name]
            with open(env, 'rb') as fp:
                env_ds = pickle.load(fp)
                self.env_dim += env_ds.metadata['dim']
                self.env_data.append(env_ds)
        print('导入原始数据成功~~~')

    def transform_data(self):
        # 首先需要把 target
        time_features = time_features_from_frequency_str(self.config.freq)
        self.time_dim = len(time_features)
        transformation = Chain([
            AddInitTimeFeature(
                start_field=FieldName.START,
                output_field=INIT_TIME,
                time_features=time_features
            ),
            SwapAxes(
                input_fields=[FieldName.TARGET],
                axes=(0,1),
            ),
            AddObservedValuesIndicator(
                target_field = FieldName.TARGET,
                output_field = FieldName.OBSERVED_VALUES,
            ),
            # 这里的 AddTimeFeatures，面对的ts, 维度应该为(features,T)
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features,
                pred_length=self.config.pred_length,
            ),
            InstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                train_sampler=TestSplitSampler(),
                past_length=self.config.past_length,
                future_length=self.config.pred_length,
                output_NTC=True,
                time_series_fields=[
                    FieldName.FEAT_TIME,
                    FieldName.OBSERVED_VALUES
                ],
                pick_incomplete=False,
            )

        ])
        print('已设置时间特征~~')
        # 设置环境变量的 dataloader
        #self.config.batch_size
        env_train_iters = [iter(TrainDataLoader_OnlyPast(
            dataset = self.env_data[i].train,
            transform = transformation,
            batch_size = self.config.batch_size,
            num_batches_per_epoch = self.config.num_batches_per_epoch,
        )) for i in range(len(self.env_data))]
        env_test_iters = [iter(InferenceDataLoader_WithFuture(
            dataset=self.env_data[i].test,
            transform=transformation,
            batch_size=self.config.batch_size,
        )) for i in range(len(self.env_data))]
        target_train_iters = [iter(TrainDataLoader_OnlyPast(
            dataset=self.target_data[i].train,
            transform=transformation,
            batch_size=self.config.batch_size,
            num_batches_per_epoch=self.config.num_batches_per_epoch,
        )) for i in range(len(self.target_data))]
        target_test_iters = [iter(InferenceDataLoader_WithFuture(
            dataset=self.target_data[i].test,
            transform=transformation,
            batch_size=self.config.batch_size,
        )) for i in range(len(self.target_data))]

        self.env_train_loader = mergeIterOut(env_train_iters,
                                             fields=[FieldName.OBSERVED_VALUES , FieldName.TARGET],
                                             include_future=True)
        self.target_train_loader = stackIterOut(target_train_iters,
                                                fields=[FieldName.OBSERVED_VALUES  , FieldName.TARGET],
                                                dim=0,
                                                include_future=False)
        self.env_test_loader = mergeIterOut(env_test_iters,
                                            fields=[FieldName.OBSERVED_VALUES , FieldName.TARGET],
                                            include_future=True)
        self.target_test_loader = stackIterOut(target_test_iters,
                                               fields=[FieldName.OBSERVED_VALUES, FieldName.TARGET ],
                                               dim=0,
                                               include_future=True)

    def build_module(self):
        with  tf.variable_scope('global_variable', reuse=tf.AUTO_REUSE):
            # 放入可能出现的SSM参数
            #  A(transition)是对角矩阵表示在转移的时候尽可能保持不变, B(control) 和 C(emission) 从高斯分布中随机进行采样
            init = np.array([np.eye(self.config.dim_l).astype(np.float32) for _ in range(self.config.K)])  # (K, dim_z , dim_z)
            init = np.tile(init, reps=(self.ssm_num, 1, 1, 1))
            A = tf.get_variable('A', initializer=init)
            # A = tf.tile(tf.expand_dims(A,0), multiples=(self.ssm_num, 1, 1, 1))  # (ssm_num , K , dim_z, dim_z)

            init = np.array([self.config.init_kf_matrices * np.random.randn(self.config.dim_l, self.config.dim_u).astype(np.float32)
                          for _ in range(self.config.K)])
            init = np.tile(init, reps=(self.ssm_num, 1, 1, 1))
            B = tf.get_variable('B' , initializer=init)
            # B = tf.tile(tf.expand_dims(B,0) , multiples=(self.ssm_num, 1, 1, 1))  # (ssm_num , K , dim_z, dim_z)

            init = np.array([self.config.init_kf_matrices * np.random.randn(self.config.dim_z, self.config.dim_l).astype(np.float32)
                          for _ in range(self.config.K)])
            init = np.tile(init, reps=(self.ssm_num, 1, 1, 1))
            C = tf.get_variable('C' , initializer=init)
            # C = tf.tile(tf.expand_dims(C,0), multiples=(self.ssm_num, 1, 1, 1))  # (ssm_num , K , dim_z, dim_z)

            # Initial observed variable z_0
            init = np.zeros((self.config.dim_z,), dtype=np.float32)
            # init = np.tile(init, reps=(self.ssm_num, 1))
            z_0 = tf.get_variable('z_0' , initializer=init)
            z_0 = tf.tile(tf.expand_dims(z_0,0) ,multiples=(self.ssm_num,1))

            # p(l_1) , SSM 隐空间的的的初始状态, mu , Sigma 分别代表 隐状态的 均值和方差
            #mu = np.zeros((self.config.batch_size, self.config.dim_l), dtype=np.float32)
            #mu = np.tile(mu, reps=(self.ssm_num, 1, 1)) #(ssm_num , bs , dim_l)
            #mu = tf.get_variable('mu' , initializer=mu)
            # mu = tf.tile(tf.expand_dims(mu, 0), multiples=(self.ssm_num, 1, 1))
            #Sigma = np.tile(self.config.init_cov * np.eye(self.config.dim_l, dtype=np.float32), (self.config.batch_size, 1, 1))
            #Sigma = np.tile(Sigma, reps=(self.ssm_num, 1, 1, 1)) #(ssm_num , bs , dim_l , dim_l)
            #Sigma = tf.get_variable('Sigma' , initializer=Sigma)
            # Sigma = tf.tile(tf.expand_dims(Sigma, 0), multiples=(self.ssm_num, 1, 1, 1))

        self.init_vars = dict(A=A, B=B, C=C, z_0=z_0)
        # self.init_vars['mu'] =mu
        # self.init_vars['Sigma'] =Sigma

        if self.config.scaling:
            self.scaler = MeanScaler(keepdims=False)
        else:
            self.scaler = NOPScaler(keepdims=False)

        with tf.variable_scope('prior_exactor', initializer=tf.contrib.layers.xavier_initializer()):
            self.prior_mean_model = tf.layers.Dense(units=self.config.dim_l,dtype=tf.float32 ,name='prior_mean')
            self.prior_cov_diag_model = tf.layers.Dense(
                units=self.config.dim_l,
                dtype=tf.float32,
                activation=tf.keras.activations.sigmoid,
                name='prior_cov'
            )

        # TODO: time_feature 是否要用两个不同的lstm ？
        with tf.variable_scope('time_feature_exactor', initializer=tf.contrib.layers.xavier_initializer() ) :
            self.time_feature_lstm = []
            for k in range(self.config.time_exact_layers):
                cell = tf.nn.rnn_cell.LSTMCell(num_units=self.config.time_exact_cells , name='time_lstm_cell_{}'.format(k))
                cell = tf.nn.rnn_cell.ResidualWrapper(cell) if k > 0 else cell
                cell = (
                    tf.nn.rnn_cell.DropoutWrapper(cell,state_keep_prob=1.0-self.config.dropout_rate)
                    if self.config.dropout_rate > 0.0
                    else cell
                )
                self.time_feature_lstm.append(cell)

            self.time_feature_lstm = tf.nn.rnn_cell.MultiRNNCell(self.time_feature_lstm)
            # self.time_feature_lstm = tf.nn.rnn_cell.BasicLSTMCell(self.time_feature_lstm)

            # 关于 noise 的大小是否要 限B定在 0 和 1 之间，也值得讨论
            self.noise_emission_model = tf.layers.Dense(units=1, dtype=tf.float32, name='noise_emission' , activation=tf.nn.softmax)
            self.noise_transition_model = tf.layers.Dense(units=1 , dtype=tf.float32 , name='noise_transition' , activation=tf.nn.softmax)

        with tf.variable_scope('env_exactor', initializer=tf.contrib.layers.xavier_initializer() ) :
            self.env_lstm = []
            for k in range(self.config.env_exact_layers):
                cell = tf.nn.rnn_cell.LSTMCell(num_units=self.config.env_exact_cells,name='env_lstm_cell_{}'.format(k))
                cell = tf.nn.rnn_cell.ResidualWrapper(cell) if k > 0 else cell
                cell = (
                    tf.nn.rnn_cell.DropoutWrapper(cell, state_keep_prob=1.0 - self.config.dropout_rate)
                    if self.config.dropout_rate > 0.0
                    else cell
                )
                self.env_lstm.append(cell)

            self.env_lstm = tf.nn.rnn_cell.MultiRNNCell(self.env_lstm)
            # 需要根据ssm的数量决定 dense
            self.u_model = [
                tf.layers.Dense(units=self.config.dim_u , dtype=tf.float32 , name='u_{}_model'.format(i))
                for i in range(self.ssm_num)
            ]


        with tf.variable_scope('alpha', initializer=tf.contrib.layers.xavier_initializer()) :
            if self.config.alpha_rnn == True:
                self.alpha_model = tf.nn.rnn_cell.LSTMCell(num_units=self.config.alpha_units ,name='alpha_model')
                self.alpha_dense = tf.layers.Dense(units=self.config.K ,
                                                   activation=tf.nn.softmax,
                                                   dtype=tf.float32 ,
                                                   name='alpha_dense')

        return self

    def alpha(self, inputs, state=None, reuse=None, name='alpha'):
        """
        是为了给SSM产生转移矩阵的混合系数
        Args:
            inputs: tensor to condition mixing vector on (ssm_num , bs , 1)
            state: previous state of RNN network
            reuse: `True` or `None`; if `True`, we go into reuse mode for this scope as
                    well as all sub-scopes; if `None`, we just inherit the parent scope reuse.
            name: name of the scope

        Returns:
            alpha: mixing vector of dimension (batch size, K)
            state: new state
            u: either inferred u from model or pass-through
        """

        # If K == 1, return inputs
        if self.config.K == 1:
            return tf.ones([self.ssm_num,self.config.batch_size, self.config.K]), state

        # Initial state for the alpha RNN
        with tf.variable_scope(name, reuse=reuse):
            # 只执行一步
            # reshape 之后确实可以恢复回来
            inputs = tf.reshape(inputs , (-1,1) ) #(bs*ssm_num ,1)
            output, state = self.alpha_model(inputs, state) #(bs*ssm_num , cell_units) ,(h,c)
            # Get Alpha as the first part of the output
            alpha = self.alpha_dense(output[:, :self.config.alpha_units])
            alpha = tf.reshape(alpha , (self.ssm_num,self.config.batch_size,self.config.K))
        return alpha, state

    def build_train_forward(self):
        train_env_norm , self.env_scale  = self.scaler.build_forward(
            data = self.placeholders['past_environment'],
            observed_indicator = self.placeholders['past_env_observed'],
            seq_axis=1
        )#(bs ,seq , env_dim) , (bs , env_dim)

        train_target_norm , self.target_scale , = self.scaler.build_forward(
            data = self.placeholders['past_target'],
            observed_indicator = self.placeholders['past_target_observed'],
            seq_axis=2
        )#(ssm_num , bs , seq , 1) , (ssm_num ,bs , 1)

        # 一定要注意 Tensor 和 Variable 千万不能随意当成相同的东西
        # 对 time_feature 进行处理
        # TODO: 现在假设 transition的噪声 emission的噪声都由 target的时间特征决定
        target_time_rnn_out, self.target_time_train_last_state = tf.nn.dynamic_rnn(
            cell=self.time_feature_lstm,
            inputs=self.placeholders['target_past_time_feature'],
            initial_state=None,
            dtype=tf.float32,
        )  # (bs,seq_length ,hidden_dim) layes x ([bs,h_dim],[bs,h_dim])
        mu = self.prior_mean_model(target_time_rnn_out[:,0,:]) #(bs,dim_l)
        mu = tf.tile(tf.expand_dims(mu,0) ,[self.ssm_num ,1,1]) #(ssm_num ,bs , dim_l)
        Sigma = self.prior_cov_diag_model(target_time_rnn_out[:,0,:])
        Sigma = tf.tile(tf.expand_dims(make_nd_diag(Sigma ,self.config.dim_l) ,0) ,[self.ssm_num ,1,1,1])
        eyes = tf.eye(self.config.dim_l)
        train_Q = self.noise_transition_model(target_time_rnn_out) #(bs, seq , 1)
        train_Q = tf.multiply(eyes , tf.expand_dims(train_Q , axis=-1))
        train_R = self.noise_emission_model(target_time_rnn_out) #(bs, seq , 1)
        train_R = tf.expand_dims(train_R , axis=-1)

        print('train network---生成噪声 Q shape : {} , R shape : {}'.format(train_Q.shape , train_R.shape))
        # 对 environment 进行处理
        env_rnn_out, self.env_train_last_state = tf.nn.dynamic_rnn(
            cell=self.env_lstm,
            inputs=train_env_norm,
            initial_state=None,
            dtype=tf.float32
        )  # (bs,seq_length ,hidden_dim)
        if self.config.use_env:
            train_u = tf.stack(
                [model(env_rnn_out)
                for model in self.u_model],
                axis = 0
            ) #(ssm_num , bs ,seq , dim_u)
        else:
            #重定向 放入KF的 global_B
            self.init_vars['B'] = tf.get_variable('B' , trainable=False , initializer=self.init_vars['B'])
            train_u = tf.constant(
                np.zeros(shape=(self.ssm_num , self.config.batch_size , self.config.past_length , self.config.dim_u),dtype=np.float32)
            )
        print('train network---由环境变量产生的 u: ' ,  train_u)

        # 为 Kalman Filter 提供LSTM的初始状态
        dummy_lstm = tf.nn.rnn_cell.LSTMCell(self.config.alpha_units)
        alpha_lstm_init = dummy_lstm.zero_state(self.config.batch_size * self.ssm_num, tf.float32)

        self.train_kf = MultiKalmanFilter(ssm_num = self.ssm_num,
                                          dim_l=self.config.dim_l,
                                          dim_z=self.config.dim_z,
                                          dim_u=self.config.dim_u,
                                          dim_k=self.config.K,
                                          A=self.init_vars['A'],  # state transition function
                                          B=self.init_vars['B'],  # control matrix
                                          C=self.init_vars['C'],  # Measurement function
                                          R=train_R,  # measurement noise
                                          Q=train_Q,  # process noise
                                          z=train_target_norm,  # output 其实
                                          u = train_u,
                                          # mask 这里面会多一维度 (bs,seq , ssm_num ,1)
                                          mask=self.placeholders['past_target_observed'],
                                          mu=mu,
                                          Sigma=Sigma,
                                          z_0=self.init_vars['z_0'],
                                          alpha_fn=self.alpha,
                                          state = alpha_lstm_init
                                          )

        # z_pred_scaled(ssm_num ,bs , seq , 1)
        # pred_l_0[(ssm_num, bs , seq ,dim_l),(ssm_num, bs , seq ,dim_l,dim_l)]
        # pred_alpha_0 (ssm_num, bs ,K)
        # filter_ll (ssm_num ,bs ,seq)
        self.z_trained_scaled , self.pred_l_0, self.pred_alpha_0 , alpha_train_states, self.filter_ll , self.train_state = self.train_kf.filter()
        self.alpha_train_last_state = tf.nn.rnn_cell.LSTMStateTuple(
            c=alpha_train_states.c[-1],
            h=alpha_train_states.h[-1]
        )

        # 计算 loss 的batch情况，但是加负号
        self.train_z_mean = tf.math.multiply(self.z_trained_scaled, tf.expand_dims(self.target_scale, 2))
        self.filter_batch_nll_mean = tf.math.reduce_mean(
            tf.math.reduce_mean(-self.filter_ll, axis=-1)
            ,axis=0
        )#(bs,)
        self.filter_nll_mean = tf.math.reduce_mean(
            self.filter_batch_nll_mean
        ) #()
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        gvs = optimizer.compute_gradients(self.filter_nll_mean)
        capped_gvs = [(tf.clip_by_value(grad, -1., 10.), var) for grad, var in gvs if grad != None]
        self.train_op = optimizer.apply_gradients(capped_gvs)

        return self

    def build_predict_forward(self):
        # 对预测域的数据先进行标准化(注：仅使用训练域的 mean 进行 scale)
        pred_env_norm = tf.math.divide(self.placeholders['pred_environment'] , tf.expand_dims(self.env_scale,1))
        # env_norm = tf.math.multiply(env_norm , self.placeholders['pred_env_observed'])
        #首先,先对time_feature进行利用
        target_time_rnn_out, _ = tf.nn.dynamic_rnn(
            cell=self.time_feature_lstm,
            inputs=self.placeholders['target_pred_time_feature'],
            initial_state=self.target_time_train_last_state,
            dtype=tf.float32,
        )  # (bs,pred ,hidden_dim)
        eyes = tf.eye(self.config.dim_l)
        pred_Q = self.noise_transition_model(target_time_rnn_out)  # (bs, pred , 1)
        pred_Q = tf.multiply(eyes, tf.expand_dims(pred_Q, axis=-1))
        pred_R = self.noise_emission_model(target_time_rnn_out)  # (bs, pred , 1)
        pred_R = tf.expand_dims(pred_R, axis=-1)
        print('pred network---生成噪声 Q shape : {} , R shape : {}'.format(pred_Q.shape , pred_R.shape))

        # 对 environment 进行处理
        env_rnn_out, _ = tf.nn.dynamic_rnn(
            cell=self.env_lstm,
            inputs=pred_env_norm,
            initial_state=self.env_train_last_state,
            dtype=tf.float32
        )  # (bs,seq_length ,hidden_dim)

        if self.config.use_env:
            self.pred_u = tf.stack(
                [model(env_rnn_out)
                 for model in self.u_model],
                axis=0
            )  # (ssm_num , bs ,pred , dim_u)
        else:
            self.pred_u = tf.constant(
                np.zeros(shape=(self.ssm_num, self.config.batch_size, self.config.pred_length, self.config.dim_u),dtype=np.float32)
            )
        print('pred network---环境变量生成的信息：' , self.pred_u)
        self.pred_kf = MultiKalmanFilter( ssm_num = self.ssm_num,
                                          dim_l=self.config.dim_l,
                                          dim_z=self.config.dim_z,
                                          dim_u=self.config.dim_u,
                                          dim_k=self.config.K,
                                          A=self.init_vars['A'],  # state transition function
                                          B=self.init_vars['B'],  # control matrix
                                          C=self.init_vars['C'],  # Measurement function
                                          R=pred_R,  # measurement noise
                                          Q=pred_Q,  # process noise
                                          u=self.pred_u,
                                          alpha_0=self.pred_alpha_0,
                                          mu=self.pred_l_0[0],
                                          Sigma=self.pred_l_0[1],
                                          alpha_fn=self.alpha,
                                          state=self.alpha_train_last_state
                                      )
        self.pred_l_mean , self.pred_l_cov, self.pred_z_mean_scaled , self.pred_z_cov = self.pred_kf.compute_forwards_pred_mode()
        self.pred_z_mean_scaled = tf.transpose(self.pred_z_mean_scaled, [1,2,0,3])
        self.pred_z_cov = tf.transpose(self.pred_z_cov , [1,2,0,3,4])
        #(pred ,ssm_num, bs, dim_z)
        #(pred , ssm_num ,bs, dim_z, dim_z)
        return self

    def initialize_variables(self):
        """ Initialize variables or load saved models
        :return: self
        """
        # Setup saver
        self.saver = tf.train.Saver()

        # Initialize or reload variables
        if self.config.reload_model is not '':
            if self.config.reload_time  == '':
                model_params = 'model_params'
            else:
                model_params = 'model_params_{}'.format(self.config.reload_time)
            params_path = os.path.join(self.config.logs_dir, self.config.reload_model, model_params)
            print("Restoring model in %s" % params_path)
            param_name = get_model_params_name(params_path)
            params_path = os.path.join(params_path, param_name)
            self.saver.restore(self.sess, params_path)
        else:
            print("Training to get new model params")
            self.sess.run(tf.global_variables_initializer())
        return self

    def train(self):
        #如果导入已经有的信息,不进行训练
        if self.config.reload_model != '':
            print('有已经训练好的参数,不需要训练~~~~~')
            return
        sess = self.sess
        log_name = '_'.join([
                            'freq(%s)'%(self.config.freq),
                            'env(%s)' % (self.config.environment) if self.config.use_env else '',
                            'lags(%d)'%(self.config.maxlags),
                           'past(%d)'%(self.config.past_length)
                            ,'pred(%d)'%(self.config.pred_length)
                            , 'u(%d)' % (self.config.dim_u)
                            , 'l(%d)' % (self.config.dim_l)
                            , 'K(%d)' % (self.config.K)
                            , 'T(%d-%d)' % (
                                 self.config.time_exact_layers, self.config.time_exact_cells)
                            , 'E(%d-%d)' % (
                                 self.config.env_exact_layers, self.config.env_exact_cells)
                            , 'α(%d)' % (
                             self.config.alpha_units)
                            , 'epoch(%d)' % (self.config.epochs)
                            , 'bs(%d)' % (self.config.batch_size)
                            , 'bn(%d)' % (self.config.num_batches_per_epoch)
                            , 'lr(%s)' % (str(self.config.learning_rate))
                            , 'initKF(%s)' % (str(self.config.init_kf_matrices))
                            , 'dropout(%s)' % (str(self.config.dropout_rate))

                         ]
                    )
        self.train_log_path = os.path.join(self.config.logs_dir,log_name)
        params_path = os.path.join(self.train_log_path , 'model_params')
        params_path = add_time_mark_to_dir(params_path)
        print('训练参数保存在 --->', params_path)

        if not os.path.isdir(self.train_log_path):
            os.makedirs(self.train_log_path)
        num_batches = self.config.num_batches_per_epoch
        best_epoch_info = {
            'epoch_no': -1,
            'filter_nll': np.Inf
        }
        train_plot_points = {}

        for epoch_no in range(self.config.epochs):
            #执行一系列操作
            #产生新的数据集
            tic = time.time()
            epoch_loss = 0.0

            with tqdm(range(num_batches)) as it:
                for batch_no in it :
                    env_batch_input  = next(self.env_train_loader)
                    target_batch_input = next(self.target_train_loader)
                    # print('第 ',batch_no,'的env_batch_input["past_target"]:',  env_batch_input['past_target'].shape)
                    feed_dict = {
                        self.placeholders['past_environment'] : env_batch_input['past_target'],
                        self.placeholders['past_env_observed'] : env_batch_input['past_%s' % (FieldName.OBSERVED_VALUES)],
                        self.placeholders['env_past_time_feature'] : env_batch_input['past_time_feat'],
                        self.placeholders['target_past_time_feature'] : target_batch_input['past_time_feat'],
                        self.placeholders['past_target'] : target_batch_input['past_target'],
                        self.placeholders['past_target_observed'] : target_batch_input['past_%s' % (FieldName.OBSERVED_VALUES)]
                    }


                    batch_nll , _   = sess.run([self.filter_batch_nll_mean , self.train_op]
                                            , feed_dict=feed_dict)
                    if epoch_no % 10 == 0 and epoch_no > 0 :
                        forward_state = sess.run(self.train_state,feed_dict = feed_dict )
                        # 查看 latent state 训练的过程
                        path = 'plot/latent_state/' + log_name + '/epoch_{}'.format(epoch_no)
                        l_mean = forward_state[0];
                        l_cov = forward_state[1];
                    # 将训练时的 output_mean 和 真实值进行比对 #(ssm_num ,bs , seq)
                    # batch_pred = sess.run(self.train_z_mean, feed_dict=feed_dict)
                    # plot_train_pred(path=self.train_log_path,targets=self.config.target, data=target_batch_input['past_target'], pred=batch_pred,
                    #                 batch=batch_no, epoch=epoch_no, plot_num=3 , plot_length=self.config.past_length
                    #                 , time_start=target_batch_input['start'], freq=self.config.freq)
                    epoch_loss += np.sum(batch_nll)
                    avg_epoch_loss = epoch_loss/((batch_no+1)*self.config.batch_size)

                    it.set_postfix(
                        ordered_dict={
                            "avg_epoch_loss":  avg_epoch_loss
                        },
                        refresh=False,
                    )
            toc = time.time()
            #将本epoch训练的结果添加到起来
            train_plot_points[epoch_no] = avg_epoch_loss
            logging.info(
                "Epoch[%d] Elapsed time %.3f seconds",
                epoch_no,
                (toc - tic),
            )

            logging.info(
                "Epoch[%d] Evaluation metric '%s'=%f",
                epoch_no,
                "epoch_loss",
                avg_epoch_loss,
            )
            logging.info(
                "Epoch[%d] the best of this training(before this epoch) '%d <==> %f'",
                epoch_no,
                best_epoch_info['epoch_no'],
                best_epoch_info['filter_nll'],
            )
            #如果当前 epoch 的结果比之前的要好，则立刻取代，保存
            if avg_epoch_loss < best_epoch_info['filter_nll']:
                del_previous_model_params(params_path)
                best_epoch_info['filter_nll'] = avg_epoch_loss
                best_epoch_info['epoch_no'] = epoch_no
                self.train_save_path = os.path.join(params_path
                                                    ,'best_{}_epoch({})_nll({})'.format(self.config.target.replace(',' ,'_'),
                                                         epoch_no,
                                                         best_epoch_info['filter_nll'])
                                                    )
                self.saver.save(sess, self.train_save_path)
        plot_train_epoch_loss(train_plot_points, self.train_log_path ,params_path.split('_')[-1])
        logging.info(
            f"Loading parameters from best epoch "
            f"({best_epoch_info['epoch_no']})"
        )

        logging.info(
            f"Final loss: {best_epoch_info['filter_nll']} "
            f"(occurred at epoch {best_epoch_info['epoch_no']})"
        )

        # save net parameters
        logging.getLogger().info("End models training")

    def predict(self):
        sess = self.sess

        # 如果是 训练之后接着的 predict 直接采用之前 train 产生的save_path
        if hasattr(self ,'train_save_path'):
            path = self.train_save_path
            try:
                self.saver.restore(sess, save_path=path)
                print('there is no problem in restoring models params')
            except:
                print('something bad appears')
            finally:
                print('whatever ! life is still fantastic !')

        eval_result_root_path = 'evaluate/results/{}_slice({})_past({})_pred({})'.format(self.config.target.replace(',' ,'_') , self.config.slice ,self.config.past_length , self.config.pred_length)
        model_result = []; ground_truth_result =[];

        if not os.path.exists(eval_result_root_path):
            os.makedirs(eval_result_root_path)
        if self.config.use_env:
            result_prefix = 'shared_ssm(%s)_'%(self.config.target)
        else:
            result_prefix = 'shared_ssm_without_env(%s)_' % (self.config.target)
        model_result_path = os.path.join(eval_result_root_path,'{}.pkl'.format(
            result_prefix
            + (self.train_log_path.split('/')[-1] if hasattr(self, 'train_log_path') else self.config.reload_model)
            + ('_num_samples(%d)'%(self.config.num_samples) if self.config.num_samples !=1 else '')
        ))
        model_result_path = add_time_mark_to_file(model_result_path)
        print('结果保存在-->',model_result_path)
        # 把处理完的数据放到 evaluate/results 里面
        ground_truth_path = os.path.join(eval_result_root_path,'{}.pkl'.format(
                'ground_truth(%s)_start(%s)_freq(%s)_past(%d)_pred(%d)'
                %(self.config.target, self.config.start , self.config.freq
                  , self.config.past_length , self.config.pred_length)
            )
        )
        print('ground_truth的保存的地方：' ,ground_truth_path)


        #不沉迷画图，把预测结果保存到pickle对象里面
        for ( target_batch ,  env_batch)in zip(
                enumerate(self.target_test_loader, start=1),
                enumerate(self.env_test_loader , start = 1)
        ):
            batch_no = target_batch[0]
            print('当前做Inference的第{}个batch的内容'.format(batch_no))
            target_batch_input = target_batch[1]
            env_batch_input = env_batch[1]
            if target_batch_input != None and env_batch_input != None:
                # 主要是为了能够获取时间的信息
                if not os.path.exists(ground_truth_path):
                    ground_truth_result.append(target_batch_input)

                # 这里要注意，因为我的batch_size被固定了，所以，这里要添加一个对数量的补全
                target_batch_complete , bs = complete_batch(batch = target_batch_input , batch_size = self.config.batch_size)
                env_batch_complete , _ = complete_batch(batch=env_batch_input, batch_size=self.config.batch_size)
                # np.zeros(env_batch_complete['past_target'].shape)  np.zeros(env_batch_complete['future_target'].shape)
                feed_dict = {
                    self.placeholders['past_environment'] :  env_batch_complete['past_target'],
                    self.placeholders['past_env_observed'] : env_batch_complete['past_observed_values'],
                    self.placeholders['env_past_time_feature'] : env_batch_complete['past_time_feat'],
                    self.placeholders['pred_environment']: env_batch_complete['future_target'],

                    self.placeholders['env_pred_time_feature'] : env_batch_complete['future_time_feat'],
                    self.placeholders['past_target'] : target_batch_complete['past_target'],
                    self.placeholders['past_target_observed'] : target_batch_complete['past_observed_values'],
                    self.placeholders['target_past_time_feature'] : target_batch_complete['past_time_feat'],
                    self.placeholders['target_pred_time_feature'] : target_batch_complete['future_time_feat']
                }

                # 这个部分只是为了方便画图，所以才把训练时候的结果也运行出来
                batch_train_z, z_scale = sess.run([self.train_z_mean[:,:bs], self.target_scale[:,:bs]], feed_dict=feed_dict)
                batch_pred_l_mean , batch_pred_l_cov , batch_pred_u = sess.run(
                    [self.pred_l_mean , self.pred_l_cov , self.pred_u] , feed_dict = feed_dict
                )
                batch_pred_z_mean_scaled, batch_pred_z_cov  = sess.run(
                    [self.pred_z_mean_scaled[:,:bs] , self.pred_z_cov[:,:bs] ], feed_dict=feed_dict
                )

                # 以下两个变量主要用于绘图
                # batch_ground_truth = np.concatenate(
                #     [target_batch_complete['past_target'] , target_batch_complete['future_target']]
                #     ,axis = 2
                # )[:,:bs]
                #  这里的mean_pred是不进行采样的
                # mean_pred = np.concatenate(
                #     [batch_train_z, np.multiply(batch_pred_z_mean_scaled, np.expand_dims(z_scale, axis=2))]
                #     , axis=2
                # )
                # 以下代码用于输出当前训练的预测图，但是现在为了比较多个算法的性能，
                # if hasattr(self , 'train_log_path'):
                #     pred_plot_path = self.train_log_path
                # else:
                #     pred_plot_path = os.path.join(self.config.logs_dir, self.config.reload_model)
                # plot_train_pred(path=pred_plot_path,targets=self.config.target,data=ground_truth
                #                 ,pred=mean_pred,epoch='__TEST__',batch=batch_no, plot_num=bs,
                #                 plot_length=self.config.pred_length*5,
                #                 time_start=target_batch_complete['start'],freq=self.config.freq
                # )

                if self.config.num_samples == 1:
                    mean_pred = np.concatenate(
                        [batch_train_z , np.multiply(batch_pred_z_mean_scaled, np.expand_dims(z_scale , axis=2))]
                        , axis=2
                    )
                    model_result.append(mean_pred)
                else:
                    scale = np.expand_dims(z_scale,axis=2)#(ssm_num ,bs , dim_z) --> #(ssm_num ,bs , 1 , dim_z)
                    batch_pred_z_mean = np.multiply(scale , batch_pred_z_mean_scaled)
                    pred_samples = samples_with_mean_cov(batch_pred_z_mean , batch_pred_z_cov , self.config.num_samples)
                    # pred_samples = np.multiply(scale , pred_samples_scale)
                    model_result.append(pred_samples)
                    pass



        model_result = np.concatenate(model_result, axis=1) #(ssm_num ,bs, seq,num_samples,1)
        with open(model_result_path , 'wb') as fp:
            pickle.dump(model_result , fp)
        if not os.path.exists(ground_truth_path):
            with open(ground_truth_path , 'wb') as fp:
                pickle.dump(ground_truth_result ,fp)

        return self


    def evaluate(self, forecast=None):
        pass



# prophet_compared code
'''
# 做training 时
print('\n----- training数据集 ------')
for batch_no, batch_data in enumerate(self.target_train_loader, start=1):
    if batch_data != None and batch_no <= self.config.num_batches_per_epoch:
        print('第{}个batch的内容init_time:{}  , past_target: {}  , past_time : {} , past_observed :{} ,'.format(
            batch_no,
            batch_data[INIT_TIME].shape,
            batch_data['past_target'].shape,
            batch_data['past_time_feat'].shape,
            batch_data['past_%s' % (FieldName.OBSERVED_VALUES)].shape
        ))
    else:
        print('第', batch_no, '内容为空')
        break
# 做inference 时
print('\n----- inference数据集 ------')
for batch_no, batch_data in enumerate(self.target_test_loader, start=1):
    if batch_data != None and batch_no <= self.config.num_batches_per_epoch:
        print('第{}个batch的内容init_time:{}  , past_target: {} , future_target: {} '
              ', past_time : {} , future_time : {}'
              ', past_observed : {} , future_observed : {}'.format(
            batch_no,
            batch_data[INIT_TIME].shape,
            batch_data['past_target'].shape, batch_data['future_target'].shape,
            batch_data['past_time_feat'].shape, batch_data['future_time_feat'].shape,
            batch_data['past_%s' % (FieldName.OBSERVED_VALUES)].shape,
            batch_data['future_%s' % (FieldName.OBSERVED_VALUES)].shape
        ))
    else:
        print('第', batch_no, '内容为空')
        break
exit()
'''