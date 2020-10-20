import logging

import tensorflow as tf
import numpy as np
import pickle
import time
import os
from tensorflow import Tensor
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.transform import (Chain ,
                               AddObservedValuesIndicator,
                               AddInitTimeFeature,
                               AddTimeFeatures,
                               SwapAxes,
                               InstanceSplitter)
from gluonts.transform.sampler import TestSplitSampler
from gluonts.dataset.field_names import FieldName
from .scaler import MeanScaler , NOPScaler
from gluonts.lzl_shared_ssm.utils import (del_previous_model_params,
                                          plot_train_epoch_loss,
                                          plot_train_pred_NoSSMnum,
                                          create_dataset_if_not_exist,
                                          complete_batch, add_time_mark_to_dir, add_time_mark_to_file,
                                          get_model_params_name)
from gluonts.lzl_shared_ssm.models.data_loader import TrainDataLoader_OnlyPast ,InferenceDataLoader_WithFuture, mergeIterOut , stackIterOut

INIT_TIME = 'init_time'
class Common_LSTM(object):
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

        create_dataset_if_not_exist(
            paths=target_path,start=self.config.start ,past_length=self.config.past_length
            , pred_length=self.config.pred_length,slice=self.config.slice
            , timestep=self.config.timestep, freq=self.config.freq
        )


        self.target_data = []
        # 由于 target 应该都是 dim = 1 只是确定有多少个 SSM 而已
        for target_name in target_path:
            target = target_path[target_name]
            with open(target, 'rb') as fp:
                target_ds = pickle.load(fp)
                assert target_ds.metadata['dim'] == 1 , 'target 序列的维度都应该为1'
                self.target_data.append(target_ds)

        self.ssm_num = len(self.target_data)
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

        self.target_train_loader = stackIterOut(target_train_iters,
                                                fields=[FieldName.OBSERVED_VALUES  , FieldName.TARGET],
                                                dim=0,
                                                include_future=False)
        self.target_test_loader = stackIterOut(target_test_iters,
                                               fields=[FieldName.OBSERVED_VALUES, FieldName.TARGET],
                                               dim=0,
                                               include_future=True)


    def __init__(self, config , sess):
        self.config = config
        self.sess = sess

        # dataset configuration
        self.freq = config.freq
        self.past_length = config.past_length
        self.pred_length = config.pred_length


        # 放入数据集
        self.load_original_data()
        self.transform_data()

        # network configuration
        self.batch_size = config.batch_size
        self.num_layers = config.num_layers
        self.num_cells = config.num_cells
        self.cell_type = config.cell_type
        self.dropout_rate = config.dropout_rate

        #prediciton configuration
        self.scaling = config.scaling


        # 开始搭建有可能Input 对应的 placeholder
        self.train_init_time = tf.placeholder(dtype=tf.float32 ,shape=[self.batch_size , self.time_dim])
        self.past_observed_values = tf.placeholder(dtype=tf.float32 , shape=[self.batch_size ,self.past_length, 1])
        self.past_time_feat = tf.placeholder(dtype=tf.float32 , shape =[self.batch_size ,self.past_length, self.time_dim])
        self.past_target = tf.placeholder(dtype=tf.float32 , shape =[self.batch_size ,self.past_length, 1])
        self.future_time_feat = tf.placeholder(dtype=tf.float32 , shape=[self.batch_size ,self.pred_length, self.time_dim])

    def build_module(self):
        with tf.variable_scope('lstm', initializer=tf.contrib.layers.xavier_initializer() , reuse=tf.AUTO_REUSE):
            self.lstm_dense = tf.layers.Dense(units=1 ,dtype=tf.float32 ,name='prior_mean')

            self.lstm = []
            for k in range(self.num_layers):
                cell = tf.nn.rnn_cell.LSTMCell(num_units=self.num_cells)
                cell = tf.nn.rnn_cell.ResidualWrapper(cell) if k > 0 else cell
                cell = (
                    tf.nn.rnn_cell.DropoutWrapper(cell,state_keep_prob=1.0-self.dropout_rate)
                    if self.dropout_rate > 0.0
                    else cell
                )
                self.lstm.append(cell)

            self.lstm = tf.nn.rnn_cell.MultiRNNCell(self.lstm)

            if self.scaling:
                self.scaler = MeanScaler(keepdims=False)
            else:
                self.scaler = NOPScaler(keepdims=False)

            return self



    def build_train_forward(self):
        self.target_norm, self.target_scale = self.scaler.build_forward(data=self.past_target
                               , observed_indicator=self.past_observed_values) #(bs,seq, 1) #(bs,1)  dim_z == 1
        #将在lstm的第一个时间步添加 0
        target_input = tf.concat(
            [tf.zeros(shape=(self.target_norm.shape[0] , 1 ,self.target_norm.shape[-1]))
            ,self.target_norm[:,:-1]]
            , axis=1
        )#(bs, seq_len , dim_z)

        if self.config.use_time_feat:
            features = tf.concat([target_input, self.past_time_feat], axis=2)  # (bs,seq_length , time_feat + dim_z)
        else:
            features = target_input

        # output 相当于使用 lstm 产生 SSM 的参数, 且只是一部分的参数
        self.lstm_train_output, self.lstm_final_state = tf.nn.dynamic_rnn(
            cell=self.lstm,
            inputs=features,
            initial_state=None,
            dtype=tf.float32
        )  # ( bs,seq_length ,hidden_dim)

        # 去掉最后一个时间步，这与 Shared_SSM中使用 p(l_t | l_(t-1))进行预测，实验设置上来说是一样的
        lstm_pred_norm = self.lstm_dense(self.lstm_train_output) #( bs ,seq, dim_z )
        lstm_pred_norm = tf.math.multiply(lstm_pred_norm , self.past_observed_values) #(bs, seq , dim_z)
        self.lstm_pred = tf.math.multiply(lstm_pred_norm , tf.expand_dims(self.target_scale , 1)) #( bs, seq, dim_z)

        if self.config.use_orig_compute_loss:
            loss = tf.squeeze(tf.math.square(self.past_target - self.lstm_pred),-1) #(bs, seq)
        else:
            loss = tf.squeeze(tf.math.square(self.target_norm - lstm_pred_norm) , -1)
        self.batch_loss = tf.math.reduce_mean(
            loss
            ,axis=-1
        ) #(bs,)
        self.loss_function = tf.math.reduce_mean(self.batch_loss) # MSE of the pred and the real data
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        gvs = optimizer.compute_gradients(self.loss_function)
        capped_gvs = [(tf.clip_by_value(grad, -1., 10.), var) for grad, var in gvs]
        self.train_op = optimizer.apply_gradients(capped_gvs)
        # tf.squeeze(observed_context, axis=-1)

        return self

    def build_predict_forward(self):
        if self.config.use_time_feat:
            input_0 = tf.concat(
                [self.future_time_feat[:,0] #(bs, time_dim)
                    , self.target_norm[:, -1]]
                , axis=1
            )  # ( bs , time_dim+dim_z)
        else:
            input_0 = self.target_norm[:,-1]

        # 第一个时间步的训练
        pred_result = tf.TensorArray(size=self.config.pred_length,dtype=tf.float32)
        i = 0;

        def cond(i , curr, state ,result):
            return tf.less(i , self.config.pred_length)

        def body(i , curr ,state, result):
            lstm_curr ,state = self.lstm(inputs=curr , state=state)
            curr = self.lstm_dense(lstm_curr)
            if self.config.use_time_feat:
                curr = tf.concat(
                    [self.future_time_feat[:,i]
                        , curr]
                    , axis=1
                )  # (bs, seq_len+1 , time_dim)
            result = result.write(i, curr)
            return i+1 , curr, state, result

        _ , _ ,_ , pred_result = tf.while_loop(cond, body,
                          loop_vars=[i, input_0 , self.lstm_final_state , pred_result])
        pred_result_norm = pred_result.stack() #(pred , bs , dim_z)
        pred_result_norm = tf.transpose(pred_result_norm, perm=[1, 0, 2])
        if self.config.use_time_feat:
            pred_result_norm = tf.expand_dims(pred_result_norm[..., -1], axis=-1)
        self.pred_result = tf.math.multiply(pred_result_norm , tf.expand_dims(self.target_scale , 1)) #( bs, seq, dim_z)
        return self


    def initialize_variables(self):
        """ Initialize variables or load saved models
        :return: self
        """
        # Setup saver
        self.saver = tf.train.Saver()

        # Initialize or reload variables
        if self.config.reload_model is not '':
            if self.config.reload_time == '':
                model_params = 'model_params'
            else:
                model_params = 'model_params_{}'.format(self.config.reload_time)
            params_path = os.path.join(self.config.logs_dir, self.config.reload_model, model_params)
            print("Restoring model in %s" % params_path)
            param_name = get_model_params_name(params_path)
            params_path = os.path.join(params_path, param_name)
            self.saver.restore(self.sess, params_path)
        else:
            print("Training to get new models params")
            self.sess.run(tf.global_variables_initializer())
        return self

    def train(self):
        #如果导入已经有的信息,不进行训练
        if self.config.reload_model != '':
            return
        sess = self.sess
        self.train_log_path = os.path.join(self.config.logs_dir,
                                   '_'.join([
                                            'freq(%s)' % (self.config.freq)
                                            ,'past(%d)'%(self.config.past_length)
                                            ,'pred(%d)'%(self.config.pred_length)
                                            , 'LSTM(%d-%d)' % (
                                                self.config.num_layers, self.config.num_cells)
                                            ,'use_time(%s)' %(str(self.config.use_time_feat))
                                             ,'loss_origin(%s)' %(str(self.config.use_orig_compute_loss))
                                            , 'epoch(%d)' % (self.config.epochs)
                                            , 'bs(%d)' % (self.config.batch_size)
                                            , 'bn(%d)' % (self.config.num_batches_per_epoch)
                                            , 'lr(%s)' % (str(self.config.learning_rate))
                                            , 'dropout(%s)' % (str(self.config.dropout_rate))
                                     ])
                                           )
        params_path = os.path.join(self.train_log_path, 'model_params')
        params_path = add_time_mark_to_dir(params_path)
        print('训练参数保存在 --->', params_path)

        if not os.path.isdir(self.train_log_path):
            os.makedirs(self.train_log_path)

        num_batches = self.config.num_batches_per_epoch
        best_epoch_info = {
            'epoch_no': -1,
            'MSE': np.Inf
        }
        train_plot_points = {}
        for epoch_no in range(self.config.epochs):
            #执行一系列操作
            #产生新的数据集
            tic = time.time()
            epoch_loss = 0.0

            with tqdm(range(num_batches)) as it:
                for batch_no in it :
                    target_batch_input = next(self.target_train_loader)
                    for i in range(self.ssm_num):
                        feed_dict = {
                            self.train_init_time : target_batch_input[INIT_TIME],
                            self.past_time_feat: target_batch_input['past_time_feat'],
                            self.past_observed_values : target_batch_input['past_%s' % (FieldName.OBSERVED_VALUES)][i] ,
                            self.past_target : target_batch_input['past_target'][i]}

                        batch_output , _   = sess.run([self.batch_loss , self.train_op]
                                                , feed_dict=feed_dict)
                        # TRAIN 画图
                        batch_pred = sess.run(self.lstm_pred , feed_dict=feed_dict)
                        # plot_train_pred_NoSSMnum(path=self.train_log_path,targets_name=self.config.target
                        #                          , data=target_batch_input['past_target'][i], pred=batch_pred,
                        #                         batch=batch_no, epoch=epoch_no, ssm_no=i,plot_num=1
                        #                          , plot_length=self.config.past_length, time_start=target_batch_input['start'],
                        #                          freq=self.config.freq)

                        epoch_loss += np.sum(batch_output)
                        avg_epoch_loss = epoch_loss/((batch_no+1)*self.config.batch_size*(i+1))


                    avg_epoch_loss = avg_epoch_loss
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
                best_epoch_info['MSE'],
            )

            if avg_epoch_loss < best_epoch_info['MSE']:
                del_previous_model_params(params_path)
                best_epoch_info['MSE'] = avg_epoch_loss
                best_epoch_info['epoch_no'] = epoch_no
                self.train_save_path = os.path.join(params_path
                                                    ,'best_{}_epoch({})_nll({})'.format(self.config.target.replace(',' ,'_'),
                                                         epoch_no,
                                                         best_epoch_info['MSE'])
                                                    )
                self.saver.save(sess, self.train_save_path)
                #'/{}_{}_best.ckpt'.format(self.dataset,epoch_no)
        plot_train_epoch_loss(train_plot_points, self.train_log_path ,params_path.split('_')[-1])
        logging.info(
            f"Loading parameters from best epoch "
            f"({best_epoch_info['epoch_no']})"
        )

        logging.info(
            f"Final loss: {best_epoch_info['MSE']} "
            f"(occurred at epoch {best_epoch_info['epoch_no']})"
        )

        # save net parameters
        logging.getLogger().info("End models training")

    def predict(self):
        sess = self.sess
        self.all_forecast_result = []
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

        eval_result_root_path = 'evaluate/results/{}_length({})_slice({})_past({})_pred({})'.format(
            self.config.target.replace(',' ,'_'), self.config.timestep , self.config.slice ,self.config.past_length , self.config.pred_length)
        model_result = [];

        if not os.path.exists(eval_result_root_path):
            os.makedirs(eval_result_root_path)
        model_result_path = os.path.join(eval_result_root_path, '{}.pkl'.format(
            'common_lstm_'
            + (self.train_log_path.split('/')[-1] if hasattr(self, 'train_log_path') else self.config.reload_model)
        ))
        model_result_path = add_time_mark_to_file(model_result_path)

        # 不沉迷画图，把预测结果保存到pickle对象里面
        for batch_no , target_batch in enumerate(self.target_test_loader):
            print('当前做Inference的第{}个batch的内容'.format(batch_no))
            if target_batch != None :
                batch_concat = []
                for i in range(self.ssm_num):
                    # TODO:这里要注意，因为我的batch_size被固定了，所以，这里要添加一个对数量的补全
                    target_batch_complete, bs = complete_batch(batch=target_batch, batch_size=self.config.batch_size)
                    feed_dict = {
                        self.train_init_time: target_batch_complete[INIT_TIME],
                        self.past_time_feat: target_batch_complete['past_time_feat'],
                        self.past_observed_values: target_batch_complete['past_%s' % (FieldName.OBSERVED_VALUES)][i],
                        self.past_target: target_batch_complete['past_target'][i],
                        self.future_time_feat : target_batch_complete['future_time_feat']
                    }

                    # 这个部分只是为了方便画图，所以才把训练时候的结果也运行出来
                    batch_pred = sess.run(
                        self.pred_result, feed_dict=feed_dict
                    )[:bs]
                    batch_concat.append(np.expand_dims(batch_pred,axis=0))
                batch_concat = np.concatenate(batch_concat, axis=0) #(ssm_num , bs , seq ,1)
                model_result.append(batch_concat)
        model_result = np.concatenate(model_result ,axis=1) #(ssm_num ,bs, seq ,1)

        with open(model_result_path, 'wb') as fp:
            pickle.dump(model_result, fp)

        return self