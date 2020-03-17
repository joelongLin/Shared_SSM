import logging

import tensorflow as tf
import numpy as np
import pickle
import time
import os
from tensorflow import Tensor
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.model.forecast import SampleForecast
from gluonts.transform import (Chain ,
                               AddObservedValuesIndicator,
                               AddInitTimeFeature,
                               AddTimeFeatures,
                               SwapAxes,
                               InstanceSplitter)
from gluonts.transform.sampler import TestSplitSampler
from gluonts.dataset.field_names import FieldName
from .scaler import MeanScaler , NOPScaler
from gluonts.lzl_shared_ssm.utils import del_previous_model_params, plot_train_epoch_loss ,plot_train_pred_NoSSMnum
from gluonts.lzl_shared_ssm.models.data_loader import TrainDataLoader_OnlyPast ,InferenceDataLoader_WithFuture, mergeIterOut , stackIterOut

INIT_TIME = 'init_time'
class Common_LSTM(object):
    def load_original_data(self):
        name_prefix = 'data_process/processed_data/{}_{}_{}_{}.pkl'
        # 导入 target 以及 environment 的数据
        ds_names = ','.join([self.config.target,self.config.environment])
        if self.config.slice == 'overlap':
            series = self.config.timestep - self.config.past_length - self.config.pred_length + 1
            print('每个数据集的序列数量为 ,', series)
        target_path = [name_prefix.format(
            name, '%s_DsSeries_%d' % (self.config.slice, series),
                           'train_%d' % self.config.past_length, 'pred_%d' % self.config.pred_length,
        ) for name in self.config.target.split(',')]


        for path in target_path:
            # 循环 path 的时候 需要 得到当前 path 对应的 ds_name
            for ds_name in ds_names.split(','):
                if ds_name in path:
                    break
            if not os.path.exists(path) :
                logging.info('there is no dataset [%s] , creating...'%ds_name)
                os.system('python data_process/preprocessing.py -d={} -t={} -p={} -s={} -n={} -f={}'
                          .format(ds_name
                              , self.config.past_length
                              , self.config.pred_length
                              , self.config.slice
                              , self.config.timestep
                              , self.config.freq))
            else:
                logging.info(' dataset [%s] was found , good~~~' % ds_name)
        self.target_data = []
        # 由于 target 应该都是 dim = 1 只是确定有多少个 SSM 而已
        for target in target_path:
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
                                                fields=[FieldName.OBSERVED_VALUES , FieldName.FEAT_TIME , FieldName.TARGET],
                                                dim=0,
                                                include_future=False)
        self.target_test_loader = stackIterOut(target_test_iters,
                                               fields=[FieldName.OBSERVED_VALUES, FieldName.FEAT_TIME,
                                                        FieldName.TARGET],
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
        target_norm, target_scale = self.scaler.build_forward(data=self.past_target
                               , observed_indicator=self.past_observed_values) #(bs,seq, 1) #(bs,1)  dim_z == 1
        #将在lstm的第一个时间步添加 0
        target_input = tf.concat(
            [tf.zeros(shape=(target_norm.shape[0] , 1 ,target_norm.shape[-1]))
            ,target_norm]
            , axis=1
        )#(ssm_num ,bs, seq_len+1 , dim_z)

        if self.config.use_time_feat:
            time_input = tf.concat(
                [tf.expand_dims(self.train_init_time, axis=1)
                , self.past_time_feat]
                , axis=1
            )  # (ssm_num , bs, seq_len+1 , time_dim)
            features = tf.concat([target_input, time_input], axis=2)  # (bs,seq_length , time_feat + dim_z)
        else:
            features = target_input
        # output 相当于使用 lstm 产生 SSM 的参数, 且只是一部分的参数
        output, lstm_final_state = tf.nn.dynamic_rnn(
            cell=self.lstm,
            inputs=features,
            initial_state=None,
            dtype=tf.float32
        )  # (bs,seq_length+1 ,hidden_dim)

        # 去掉最后一个时间步，这与 Shared_SSM中使用 p(l_t | l_(t-1))进行预测，实验设置上来说是一样的
        lstm_pred = self.lstm_dense(output)[:,:-1,:] #(ssm_num , bs ,seq, dim_z )
        lstm_pred = tf.math.multiply(lstm_pred , self.past_observed_values) #(bs, seq , dim_z)
        self.lstm_pred = tf.math.multiply(lstm_pred , tf.expand_dims(target_scale , 1)) #( bs, seq, dim_z)

        if self.config.use_orig_compute_loss:
            loss = tf.squeeze(tf.math.square(self.past_target - self.lstm_pred),-1) #(bs, seq)
        else:
            loss = tf.squeeze(tf.math.square(target_norm - lstm_pred) , -1)
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

        return self


    def initialize_variables(self):
        """ Initialize variables or load saved models
        :return: self
        """
        # Setup saver
        self.saver = tf.train.Saver()

        # Initialize or reload variables
        if self.config.reload_model is not '':
            print("Restoring models in %s" % self.config.reload_model)
            params_path = os.path.join(self.config.reload_model , "model_params")
            param_name = os.listdir(params_path)[0].split('.')[0]
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
        self.log_path = os.path.join(self.config.logs_dir,
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
        if not os.path.isdir(self.log_path):
            os.makedirs(self.log_path)

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
                            self.past_time_feat: target_batch_input['past_time_feat'][i],
                            self.past_observed_values : target_batch_input['past_%s' % (FieldName.OBSERVED_VALUES)][i] ,
                            self.past_target : target_batch_input['past_target'][i]}

                        batch_output , _   = sess.run([self.batch_loss , self.train_op]
                                                , feed_dict=feed_dict)
                        batch_pred = sess.run(self.lstm_pred , feed_dict=feed_dict)
                        plot_train_pred_NoSSMnum(path=self.log_path, data=target_batch_input['past_target'][i], pred=batch_pred,
                                        batch=batch_no, epoch=epoch_no, ssm_no=i,plot_num=4
                                        , time_start=target_batch_input['start'], freq=self.config.freq)
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
                del_previous_model_params(self.log_path)
                best_epoch_info['MSE'] = avg_epoch_loss
                best_epoch_info['epoch_no'] = epoch_no
                self.save_path = os.path.join(self.log_path, "model_params"
                                              ,'best_{}_epoch({})_nll({})'.format(self.config.target.replace(',' ,'_'),
                                                         epoch_no,
                                                         best_epoch_info['MSE'])
                                              )
                self.saver.save(sess, self.save_path)
                #'/{}_{}_best.ckpt'.format(self.dataset,epoch_no)
        plot_train_epoch_loss(train_plot_points, self.log_path)
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
        if hasattr(self ,'save_path'):
            path = self.save_path
            try:
                self.saver.restore(sess, save_path=path)
                print('there is no problem in restoring models params')
            except:
                print('something bad appears')
            finally:
                print('whatever ! life is still fantastic !')
        iteration = len(self.test_data)//self.batch_size+1 \
                    if len(self.test_data)%self.batch_size!=0 \
                    else len(self.test_data)//self.batch_size
        for batch_no in range(iteration):
            test_input , other_info = next(self.test_iter)
            placeholders = [self.feat_static_cat, self.past_observed_values
                , self.past_seasonal_indicators, self.past_time_feat, self.past_target
                , self.future_seasonal_indicators , self.future_time_feat]
            feed_dict = dict(zip(placeholders, test_input))

            test_output = sess.run(self.predict_result,feed_dict =  feed_dict)

            outputs = [
                np.concatenate(s)[:self.num_sample_paths]
                for s in zip(test_output)
            ]
            print('当前正在处理第 %d 个 batch_no 的结果' % (batch_no))
            for i , output in enumerate(outputs):
                # 最后一个不满 batch_size 的放弃它

                sample_forecast = SampleForecast(
                    samples = output,
                    start_date = other_info[0][i],
                    freq = self.config.freq,
                )
                # if batch_no < len(self.test_data) // self.batch_size :
                #     self.all_forecast_result.append(sample_forecast)
                if batch_no*self.batch_size + i < len(self.test_data):
                    self.all_forecast_result.append(sample_forecast)
                else:
                    print('%d batch %d sample was complete , throw away' %(batch_no,i))
                    break


        return self


    def evaluate(self, forecast=None):
        ground_truth_loader = GroundTruthLoader(config=self.config)
        ground_truth = ground_truth_loader.get_ground_truth()
        if forecast is None:
            forecast = self.all_forecast_result
        finance_eval = evaluate_up_down(ground_truth ,forecast)
        if hasattr(self , 'writer_path'):
            eval_path = os.path.join(self.log_path, "metrics.json")
        else:
            eval_path = os.path.join(self.config.reload_model , "metrics.json")
        with open(eval_path, 'w') as f:
            json.dump(finance_eval, f, indent=4)
        # plot_length = self.config.past_length + self.config.pred_length
        # for i in range(321):#对于electricity来说，总共由321列
        #     plot_prob_forecasts(ground_truth[i] , forecast[i] ,self.config.dataset,i,plot_length)
        #
        # exit()
        # evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
        # agg_metrics, item_metrics = evaluator(iter(ground_truth)
        #                                       , iter(forecast)
        #                                       , num_series=len(ground_truth_loader.ds.test))
        #
        # print(json.dumps(agg_metrics, indent=4))
        pass


def plot_prob_forecasts(ts_entry, forecast_entry,ds_name ,no ,plot_length):
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.savefig('pic/{}_result/result_output_tf_{}.png'.format(ds_name,no))
    plt.close(fig)



def evaluate_up_down(ground_truth , mc_forecast):
    # 0代表跌 1代表涨
    ground_truth_labels = [1 if truth.iloc[-1].values[0] > truth.iloc[-2].values[0] else 0 for truth in ground_truth]
    forecast_labels = [1 if mc_forecast[i].mean[0] > ground_truth[i].iloc[-2].values[0] else 0
                       for i in range(len(mc_forecast))]
    acc = accuracy_score(ground_truth_labels , forecast_labels)
    return {'acc' : acc}