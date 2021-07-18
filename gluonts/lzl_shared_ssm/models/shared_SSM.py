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

matplotlib.use('Agg') 
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

INIT_TIME = 'init_time'


class SharedSSM(object):
    def __init__(self, config , sess):
        self.config = config
        self.sess = sess


        # Load Origin Data 
        self.load_original_data()
        # put data into transformer and loader
        self.transform_data()
        # placeholder
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
        
        if self.config.slice == 'overlap':
            series = self.config.timestep - self.config.past_length - self.config.pred_length + 1
            print('Dataset series num ,', series)
        elif self.config.slice == 'nolap':
            series = self.config.timestep // (self.config.past_length + self.config.pred_length)
            print('Dataset series num ,', series)
        else:
            series = 1
            print('SINGLE SERIE')
        
        # change "_" in self.config.start
        self.config.start = self.config.start.replace("_", " ")
        
        # target series
        target_path = {name:name_prefix.format(
            '%s_start(%s)_freq(%s)'%(name, self.config.start ,self.config.freq), '%s_DsSeries_%d' % (self.config.slice, series),
            'train_%d' % self.config.past_length, 'pred_%d' % self.config.pred_length,
        ) for name in self.config.target.split(',')}
        target_start = pd.Timestamp(self.config.start,freq=self.config.freq)
        env_start = target_start - self.config.maxlags * target_start.freq
        env_start = env_start.strftime(time_format_from_frequency_str(self.config.freq))
        print('environment start time：', env_start)
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
        # Know number of SSM
        for target_name in target_path:
            target = target_path[target_name]
            with open(target, 'rb') as fp:
                target_ds = pickle.load(fp)
                assert target_ds.metadata['dim'] == 1 , 'dimension of target series should be 1'
                self.target_data.append(target_ds)

        self.ssm_num = len(self.target_data)
        self.env_dim = 0
        for env_name in env_path:
            env =  env_path[env_name]
            with open(env, 'rb') as fp:
                env_ds = pickle.load(fp)
                self.env_dim += env_ds.metadata['dim']
                self.env_data.append(env_ds)
        print('Load Origin Data success~~~')

    def transform_data(self):
        
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
            # Dimension should be (features,T)
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
        print('temporal feature is set~~')
        
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
            # initial of matrix A B C
            init = np.array([np.eye(self.config.dim_l).astype(np.float32) for _ in range(self.config.K)])  # (K, dim_z , dim_z)
            init = np.tile(init, reps=(self.ssm_num, 1, 1, 1))
            A = tf.get_variable('A', initializer=init)
            

            init = np.array([self.config.init_kf_matrices * np.random.randn(self.config.dim_l, self.config.dim_u).astype(np.float32)
                          for _ in range(self.config.K)])
            init = np.tile(init, reps=(self.ssm_num, 1, 1, 1))
            B = tf.get_variable('B' , initializer=init)
            

            init = np.array([self.config.init_kf_matrices * np.random.randn(self.config.dim_z, self.config.dim_l).astype(np.float32)
                          for _ in range(self.config.K)])
            init = np.tile(init, reps=(self.ssm_num, 1, 1, 1))
            C = tf.get_variable('C' , initializer=init)
            

            # Initial observed variable z_0
            init = np.zeros((self.config.dim_z,), dtype=np.float32)
            # init = np.tile(init, reps=(self.ssm_num, 1))
            z_0 = tf.get_variable('z_0' , initializer=init)
            z_0 = tf.tile(tf.expand_dims(z_0,0) ,multiples=(self.ssm_num,1))

            

        self.init_vars = dict(A=A, B=B, C=C, z_0=z_0)

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
            # SSM number decide the layer of dense
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

        # NOTE:  Tensor and Variable are different things
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

        print('train network--- Q shape : {} , R shape : {}'.format(train_Q.shape , train_R.shape))
        # environment 
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
            # without env
            self.init_vars['B'] = tf.get_variable('B' , trainable=False , initializer=self.init_vars['B'])
            train_u = tf.constant(
                np.zeros(shape=(self.ssm_num , self.config.batch_size , self.config.past_length , self.config.dim_u),dtype=np.float32)
            )

        # LSTM initial state
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
                                          z=train_target_norm,  # output
                                          u = train_u,
                                          mask=self.placeholders['past_target_observed'],
                                          mu=mu,
                                          Sigma=Sigma,
                                          z_0=self.init_vars['z_0'],
                                          alpha_fn=self.alpha,
                                          state = alpha_lstm_init
                                          )

        # DEMENSION:
        # z_pred_scaled(ssm_num ,bs , seq , 1)
        # pred_l_0[(ssm_num, bs , seq ,dim_l),(ssm_num, bs , seq ,dim_l,dim_l)]
        # pred_alpha_0 (ssm_num, bs ,K)
        # filter_ll (ssm_num ,bs ,seq)
        self.z_trained_scaled , self.pred_l_0, self.pred_alpha_0 , alpha_train_states, self.filter_ll , self.train_state = self.train_kf.filter()
        self.alpha_train_last_state = tf.nn.rnn_cell.LSTMStateTuple(
            c=alpha_train_states.c[-1],
            h=alpha_train_states.h[-1]
        )

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
        # normalization
        pred_env_norm = tf.math.divide(self.placeholders['pred_environment'] , tf.expand_dims(self.env_scale,1))
        # env_norm = tf.math.multiply(env_norm , self.placeholders['pred_env_observed'])
        
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
        print('pred network--- Q shape : {} , R shape : {}'.format(pred_Q.shape , pred_R.shape))

        
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
        self.pred_z_mean_scaled = tf.transpose(self.pred_z_mean_scaled, [1,2,0,3]) #(pred ,ssm_num, bs, dim_z)
        self.pred_z_cov = tf.transpose(self.pred_z_cov , [1,2,0,3,4]) #(pred , ssm_num ,bs, dim_z, dim_z)
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
        if self.config.reload_model != '':
            print('Get trained model ~~~~~')
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
        print('training model stored in --->', params_path)

        if not os.path.isdir(self.train_log_path):
            os.makedirs(self.train_log_path)
        num_batches = self.config.num_batches_per_epoch
        best_epoch_info = {
            'epoch_no': -1,
            'filter_nll': np.Inf
        }
        train_plot_points = {}

        for epoch_no in range(self.config.epochs):
            
            tic = time.time()
            epoch_loss = 0.0

            with tqdm(range(num_batches)) as it:
                for batch_no in it :
                    env_batch_input  = next(self.env_train_loader)
                    target_batch_input = next(self.target_train_loader)

                    # To the reviewer#2 adding noise to the background information
                    if self.config.env_noise !=0:
                        env_batch_input['past_target'] = env_batch_input['past_target'] \
                                + np.random.normal(loc = self.config.env_noise_mean, scale = self.config.env_noise, size =(self.config.batch_size, self.config.past_length, self.env_dim))

                    
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
                        #  latent state training epoch
                        path = 'plot/latent_state/' + log_name + '/epoch_{}'.format(epoch_no)
                        l_mean = forward_state[0];
                        l_cov = forward_state[1];
                    
                    epoch_loss += np.sum(batch_nll)
                    avg_epoch_loss = epoch_loss/((batch_no+1)*self.config.batch_size)

                    it.set_postfix(
                        ordered_dict={
                            "avg_epoch_loss":  avg_epoch_loss
                        },
                        refresh=False,
                    )
            toc = time.time()
            
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

        if hasattr(self ,'train_save_path'):
            path = self.train_save_path
            try:
                self.saver.restore(sess, save_path=path)
                print('there is no problem in restoring models params')
            except:
                print('something bad appears')
            finally:
                print('whatever ! life is still fantastic !')

        # Prediction Result
        eval_result_root_path = 'evaluate/results{}/{}_length({})_slice({})_past({})_pred({})'.format(
            "" if self.config.env_noise==0 else "_noise",
            self.config.target.replace(',' ,'_'), self.config.timestep , self.config.slice ,self.config.past_length , self.config.pred_length)
        if not os.path.exists(eval_result_root_path):
            os.makedirs(eval_result_root_path)
        model_result = []; ground_truth_result =[];

        # background and latent state
        eval_analysis_root_path = 'evaluate/analysis{}/{}_length({})_slice({})_past({})_pred({})'.format(
            "" if self.config.env_noise==0  else "_noise",
            self.config.target.replace(',' ,'_') ,self.config.timestep , self.config.slice ,self.config.past_length , self.config.pred_length)
        if not os.path.exists(eval_analysis_root_path):
            os.makedirs(eval_analysis_root_path)
        prediction_analysis = {'l_mean':[] , 'l_cov':[] ,'control':[] ,'ground_truth':[] ,'z_mean_scaled':[]}
        

        
        
        # Prediction result path
        if self.config.use_env:
            result_prefix = 'shared_ssm_'
        else:
            result_prefix = 'without_env_' 
        model_result_path = os.path.join(eval_result_root_path,'{}.pkl'.format(
            result_prefix
            + (self.train_log_path.split('/')[-1] if hasattr(self, 'train_log_path') else self.config.reload_model)
            + ('_num_samples(%d)'%(self.config.num_samples) if self.config.num_samples !=1 else '' )
            + ('_env_noise(%2f)'%(self.config.env_noise) if self.config.env_noise != 0 else '')
            + ('_mean_noise(%2f)'%(self.config.env_noise_mean) if self.config.env_noise_mean != 0 else '')
        ))
        model_result_path = add_time_mark_to_file(model_result_path)
        print('Prediction result store in-->',model_result_path)
        # 设置 ground truth 的地址
        ground_truth_path = os.path.join(eval_result_root_path,
                'ground_truth_start({})_freq({})_past({})_pred({}).pkl'.format(
                self.config.start , self.config.freq
                  , self.config.past_length , self.config.pred_length)
            
        )
        print('ground truth store in ' ,ground_truth_path)


        #不沉迷画图，把预测结果保存到pickle对象里面
        for ( target_batch ,  env_batch)in zip(
                enumerate(self.target_test_loader, start=1),
                enumerate(self.env_test_loader , start = 1)
        ):
            batch_no = target_batch[0]
            print('No.{} batch are predicting'.format(batch_no))
            target_batch_input = target_batch[1]
            env_batch_input = env_batch[1]
            if target_batch_input != None and env_batch_input != None:
                # no duplicate ground truth
                if not os.path.exists(ground_truth_path):
                    ground_truth_result.append(target_batch_input)

                # complete the num as batch size
                target_batch_complete , bs = complete_batch(batch = target_batch_input , batch_size = self.config.batch_size)
                env_batch_complete , _ = complete_batch(batch=env_batch_input, batch_size=self.config.batch_size)
                
                # To the reviewer#2 we add noise to the env at the prediction range
                if self.config.env_noise !=0:
                        env_batch_complete['past_target'] = env_batch_complete['past_target'] \
                                + np.random.normal(loc = self.config.env_noise_mean, scale = self.config.env_noise, size =(self.config.batch_size, self.config.past_length, self.env_dim))
                        env_batch_complete['future_target'] =  env_batch_complete['future_target'] \
                                + np.random.normal(loc = self.config.env_noise_mean, scale = self.config.env_noise, size =(self.config.batch_size, self.config.pred_length, self.env_dim))
                
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

                
                batch_train_z, z_scale = sess.run([self.train_z_mean[:,:bs], self.target_scale[:,:bs]], feed_dict=feed_dict)
                batch_pred_l_mean , batch_pred_l_cov , batch_pred_u = sess.run(
                    [self.pred_l_mean[:,:,:bs] , self.pred_l_cov[:,:,:bs] , self.pred_u[:,:bs]] , feed_dict = feed_dict
                )
                batch_pred_z_mean_scaled, batch_pred_z_cov  = sess.run(
                    [self.pred_z_mean_scaled[:,:bs] , self.pred_z_cov[:,:bs] ], feed_dict=feed_dict
                )

              
                
                prediction_analysis['l_mean'].append(
                    np.transpose(batch_pred_l_mean ,[1,2,0,3])
                ) #(ssm, bs, seq , dim_l)
                prediction_analysis['l_cov'].append(
                    np.transpose(batch_pred_l_cov , [1,2,0,3,4])
                )
                prediction_analysis['control'].append(batch_pred_u)#(ssm, bs, seq, dim_u)
                ground_truth_scaled = np.divide(target_batch_complete['future_target'][:,:bs], np.expand_dims(z_scale , axis=2))
                prediction_analysis['ground_truth'].append(ground_truth_scaled)
                prediction_analysis['z_mean_scaled'].append(batch_pred_z_mean_scaled)


                
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

        # analysis result (analysis/)
        for name , item in prediction_analysis.items():
            item = np.concatenate(item , axis=1)
            prediction_analysis[name] = item
        model_analysis_path = os.path.join(eval_analysis_root_path , 'seq({})_dim_l({})_dim_u({})_lag({}){}.pkl'.format(
            self.config.pred_length ,self.config.dim_l , self.config.dim_u ,self.config.maxlags, '' if self.config.reload_time == '' else '_time({})'.format(self.config.reload_time)
            )
        )
        model_analysis_path = add_time_mark_to_file(model_analysis_path)
        with open(model_analysis_path , 'wb') as fp:
            pickle.dump(prediction_analysis, fp)
        
        
        

        # prediction result (results/)
        model_result = np.concatenate(model_result, axis=1) #(ssm_num ,bs, seq,num_samples,1)
        with open(model_result_path , 'wb') as fp:
            pickle.dump(model_result , fp)
        if not os.path.exists(ground_truth_path):
            with open(ground_truth_path , 'wb') as fp:
                pickle.dump(ground_truth_result ,fp)

        return self

