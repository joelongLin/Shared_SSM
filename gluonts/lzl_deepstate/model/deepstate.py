import logging

import tensorflow as tf
import numpy as np
import pickle
import time
import os
from .issm import CompositeISSM
from ..utils import make_nd_diag,weighted_average
from tensorflow import Tensor
from .lds import LDSArgsProj ,LDS
from .scaler import MeanScaler,NOPScaler
from .data_loader import DataLoader,GroundTruthLoader
from tqdm import tqdm
from .forecast import SampleForecast
from .evaluation import Evaluator
import json
import matplotlib.pyplot as plt
from ...time_feature import (TimeFeature ,time_features_from_frequency_str)

class FeatureEmbedder(object):
    def __init__(self,cardinalities,embedding_dims):
        assert (
                len(cardinalities) > 0
        ), "Length of `cardinalities` list must be greater than zero"
        assert len(cardinalities) == len(
            embedding_dims
        ), "Length of `embedding_dims` and `embedding_dims` should match"
        assert all(
            [c > 0 for c in cardinalities]
        ), "Elements of `cardinalities` should be > 0"
        assert all(
            [d > 0 for d in embedding_dims]
        ), "Elements of `embedding_dims` should be > 0"

        self.__num_features = len(cardinalities)

        def create_embedding(i: int, c: int, d: int) :
            embedding = tf.get_variable(name=f"cat_{i}_embedding" , shape=[c,d])
            return embedding

        with tf.variable_scope('feature_embedder'):
            self.__embedders = [
                create_embedding(i, c, d)
                for i, (c, d) in enumerate(zip(cardinalities, embedding_dims))
            ]

    def build_forward(self, features):
        '''
        :param features: Categorical features with shape: dynamic (N,T,C) or static (N,C), where C is the
            number of categorical features.
        :return: concatenated_tensor: Tensor
            Concatenated tensor of embeddings whth shape: (N,T,C) or (N,C),
            where C is the sum of the embedding dimensions for each categorical
            feature, i.e. C = sum(self.config.embedding_dims).
        '''
        if self.__num_features > 1:
            # we slice the last dimension, giving an array of length self.__num_features with shape (N,T) or (N)
            cat_feature_slices = tf.split(
                features, num_or_size_splits=self.__num_features,axis=-1
            )
        else:
            # F.split will iterate over the second-to-last axis if the last axis is one
            cat_feature_slices = [features]

        return tf.concat(
            [
                tf.nn.embedding_lookup(embed ,tf.dtypes.cast(tf.squeeze(cat_feature_slice, axis=-1) ,dtype=tf.int32))
                for embed, cat_feature_slice in zip(
                    self.__embedders, cat_feature_slices
                )
            ],
            axis=-1,
        )

class DeepStateNetwork(object):
    def __init__(self, config , sess):
        self.config = config
        self.sess = sess

        # dataset configuration
        self.dataset = config.dataset
        self.freq = config.freq
        self.past_length = config.past_length
        self.prediction_length = config.prediction_length
        self.add_trend = config.add_trend


        # 放入数据集
        with open('data/train_{}_{}_{}.pkl'.format(self.dataset , self.past_length,self.prediction_length) , 'rb') as fp:
            self.train_data = pickle.load(fp)
        with open('data/test_{}_{}_{}.pkl'.format(self.dataset , self.past_length,self.prediction_length) , 'rb') as fp:
            self.test_data = pickle.load(fp)
        shape_list = []
        for input in self.train_data[0]:
            shape_list.append(input.shape)

        # 将数据集放入 DataLoader 中产生一个 iterator
        self.train_iter =  DataLoader(self.train_data, self.config).data_iterator(is_train=True)
        self.test_iter = DataLoader(self.test_data , self.config).data_iterator(is_train=False)


        # network configuration
        self.batch_size = config.batch_size
        self.num_layers = config.num_layers
        self.num_cells = config.num_cells
        self.cell_type = config.cell_type
        self.dropout_rate = config.dropout_rate

        #prediciton configuration
        self.num_sample_paths = config.num_eval_samples
        self.scaling = config.scaling
        self.use_feat_dynamic_real = config.use_feat_dynamic_real
        self.use_feat_static_cat = config.use_feat_static_cat
        self.cardinality = [int(num) for num in config.cardinality.split(',')] if self.use_feat_static_cat else [1]
        embed_split = config.embedding_dimension.split(',')
        self.embedding_dimension = [int(num) for num in embed_split] if len(embed_split) > 1 \
            else [min(50, (cat + 1) // 2) for cat in self.cardinality]

        self.issm = CompositeISSM.get_from_freq(self.freq, self.add_trend)

        self.univariate = self.issm.output_dim() == 1


        # 开始搭建有可能Input 对应的 placeholder
        self.feat_static_cat = tf.placeholder(dtype=tf.float32 , shape=[self.batch_size , shape_list[0][0]])
        self.past_observed_values = tf.placeholder(dtype=tf.float32 , shape=[self.batch_size ,self.past_length, shape_list[1][1]])
        self.past_seasonal_indicators = tf.placeholder(dtype=tf.float32 , shape = [self.batch_size,self.past_length , shape_list[2][1]])
        self.past_time_feat = tf.placeholder(dtype=tf.float32 , shape =[self.batch_size ,self.past_length, shape_list[3][1]])
        self.past_target = tf.placeholder(dtype=tf.float32 , shape =[self.batch_size ,self.past_length, shape_list[4][1]])
        self.future_seasonal_indicators = tf.placeholder(dtype=tf.float32 , shape=[self.batch_size ,self.prediction_length, shape_list[2][1]])
        self.future_time_feat = tf.placeholder(dtype=tf.float32 , shape=[self.batch_size ,self.prediction_length, shape_list[3][1]])

    def build_module(self):
        with tf.variable_scope('deepstate', initializer=tf.contrib.layers.xavier_initializer() , reuse=tf.AUTO_REUSE):
            self.prior_mean_model = tf.layers.Dense(units=self.issm.latent_dim(),dtype=tf.float32 ,name='prior_mean')

            self.prior_cov_diag_model = tf.layers.Dense(
                units=self.issm.latent_dim(),
                dtype=tf.float32,
                activation=tf.keras.activations.sigmoid,  # TODO: puot explicit upper bound
                name='prior_cov'
            )

            self.lds_proj = LDSArgsProj(output_dim=self.issm.output_dim())

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

            self.embedder = FeatureEmbedder(
                # cardinalities=cardinality,
                cardinalities=self.cardinality,
                embedding_dims=self.embedding_dimension,
            )

            # print(self.embedder.)
            # exit()

            if self.scaling:
                self.scaler = MeanScaler(keepdims=False)
            else:
                self.scaler = NOPScaler(keepdims=False)

            return self

    # 将在 build_forward 里面进行调用
    def compute_lds(
        self,
        feat_static_cat: Tensor,
        seasonal_indicators: Tensor,
        time_feat: Tensor,
        length: int,
        prior_mean = None, # can be None
        prior_cov = None, # can be None
        lstm_begin_state = None, # can be None
    ):
        # embed categorical features and expand along time axis

        embedded_cat = self.embedder.build_forward(feat_static_cat)

        repeated_static_features = tf.tile(
            tf.expand_dims(embedded_cat, axis=1),
            multiples=[1,length,1]
        ) #(bs , seq_length , embedding)


        # construct big features tensor (context)
        features = tf.concat([time_feat, repeated_static_features], axis=2)#(bs,seq_length , time_feat + embedding)

        # output 相当于使用 lstm 产生 SSM 的参数
        output , lstm_final_state = tf.nn.dynamic_rnn(
            cell = self.lstm,
            inputs = features,
            initial_state = lstm_begin_state,
            dtype = tf.float32
        ) #(bs,seq_length ,hidden_dim)


        if prior_mean is None:
            prior_input = tf.squeeze(
                tf.slice(output, begin=[0, 0, 0], size=[-1, 1, -1]),
                axis=1
            ) #选用lstm 第一个时间步的结果

            prior_mean = self.prior_mean_model(prior_input)#(bs,latent_dim)
            prior_cov_diag = self.prior_cov_diag_model(prior_input)
            prior_cov = make_nd_diag( prior_cov_diag, self.issm.latent_dim())#(bs ,latent_dim ,latent_dim)


        emission_coeff, transition_coeff, innovation_coeff = self.issm.get_issm_coeff(
            seasonal_indicators
        )

        # emission_coeff（bs , seq_length , output_dim , latent_dim）
        # transition_coeff(bs, seq_length, latent_dim ,latent_dim)
        # innovation_coeff(bs ,seq_length ,latent_dim)


        noise_std, innovation, residuals = self.lds_proj.build_forward(output)
        # noise_std（bs , seq_length , 1 ）
        # innovation(bs, seq_length,  1 )
        # residuals(bs ,seq_length ,output_dim)



        lds = LDS(
            emission_coeff=emission_coeff,
            transition_coeff=transition_coeff,
            innovation_coeff=tf.math.multiply(innovation, innovation_coeff),
            noise_std=noise_std,
            residuals=residuals,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            latent_dim=self.issm.latent_dim(),
            output_dim=self.issm.output_dim(),
            seq_length=length,
        )

        return lds, lstm_final_state


    def build_train_forward(self):
        lds, _ = self.compute_lds(
            feat_static_cat=self.feat_static_cat,
            seasonal_indicators=tf.slice(
                self.past_seasonal_indicators
                ,begin=[0,0,0],size=[-1,-1,-1]
            ),
            time_feat=tf.slice(self.past_time_feat
                , begin=[0, 0, 0], size=[-1, -1, -1]
            ),
            length=self.past_length,
        )

        _, scale = self.scaler.build_forward(data=self.past_target
                               , observed_indicator=self.past_observed_values)
        #(bs,output_dim)


        observed_context = tf.slice(
            self.past_observed_values,
             begin=[0,0,0], size=[-1,-1,-1]
        )#(bs,seq_length , 1)


        ll, _, _ = lds.log_prob(
            x=tf.slice(
                self.past_target,
                begin=[0,0,0], size=[-1,-1,-1]
            ),#(bs,seq_length ,time_feat)
            observed=tf.math.reduce_min(observed_context ,axis=-1), #(bs ,seq_length)
            scale=scale,
        )#(bs,seq_length)


        self.train_result = weighted_average(
            x=-ll, axis=1, weights= tf.math.reduce_min(observed_context, axis=-1)
        )#(bs,)

        self.train_result_mean = tf.math.reduce_mean(self.train_result)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        gvs = optimizer.compute_gradients(self.train_result_mean)
        capped_gvs = [(tf.clip_by_value(grad, -1., 10.), var) for grad, var in gvs]
        self.train_op = optimizer.apply_gradients(capped_gvs)
        # tf.squeeze(observed_context, axis=-1)

        return self

    def build_predict_forward(self):
        lds, lstm_state = self.compute_lds(
            feat_static_cat=self.feat_static_cat,
            seasonal_indicators=tf.slice(
                self.past_seasonal_indicators
                ,begin=[0,0,0],size=[-1,-1,-1]
            ),
            time_feat=tf.slice(self.past_time_feat
                , begin=[0, 0, 0], size=[-1, -1, -1]
            ),
            length=self.past_length,
        )



        _, scale = self.scaler.build_forward(self.past_target, self.past_observed_values)

        observed_context = tf.slice(
            self.past_observed_values,
             begin=[0,0,0], size=[-1,-1,-1]
        )#(bs,seq_length , 1)

        _, final_mean, final_cov = lds.log_prob(
            x=tf.slice(
                self.past_target,
                begin=[0, 0, 0], size=[-1, -1, -1]
            ),  # (bs,seq_length ,time_feat)
            observed=tf.math.reduce_min(observed_context, axis=-1),  # (bs ,seq_length)
            scale=scale,
        )

        # print('lstm final state shape : ',
        #       [{'c_shape': state.c.shape, 'h_shape': state.h.shape} for state in lstm_state])

        lds_prediction, _ = self.compute_lds(
            feat_static_cat=self.feat_static_cat,
            seasonal_indicators=self.future_seasonal_indicators,
            time_feat=self.future_time_feat,
            length=self.prediction_length,
            lstm_begin_state=lstm_state,
            prior_mean=final_mean,
            prior_cov=final_cov,
        )



        samples = lds_prediction.sample(
            num_samples=self.num_sample_paths, scale=scale
        )# (num_samples, batch_size, seq_length, obs_dim)


        # (batch_size, num_samples, prediction_length, target_dim)
        #  squeeze last axis in the univariate case (batch_size, num_samples, prediction_length)
        if self.univariate:
            self.predict_result = tf.squeeze(tf.transpose(samples , [1, 0, 2, 3]),axis=3)
        else:
            self.predict_result = tf.transpose(samples , [1,0,2,3])

        return self


    def initialize_variables(self):
        """ Initialize variables or load saved model
        :return: self
        """
        # Setup saver
        self.saver = tf.train.Saver()

        # Initialize or reload variables
        if self.config.reload_model is not '':
            print("Restoring model in %s" % self.config.reload_model)
            self.saver.restore(self.sess, self.config.reload_model)
        else:
            self.sess.run(tf.global_variables_initializer())
        return self

    def train(self):
        sess = self.sess
        writer_path =os.path.join( self.config.logs_dir ,
                   '_'.join(time.asctime(time.localtime(time.time())).split(' ')[1:5]))

        writer = tf.summary.FileWriter(writer_path, sess.graph)
        num_batches = self.config.num_batches_per_epoch
        best_epoch_info = {
            'epoch_no': -1,
            'metric_value': np.Inf
        }
        # vars = tf.trainable_variables()
        # for v in vars:
        #     print(v)
        # exit()

        for epoch_no in range(self.config.epochs):
            #执行一系列操作
            #产生新的数据集
            tic = time.time()
            epoch_loss = 0.0


            with tqdm(range(num_batches)) as it:
                for batch_no in it :
                    batch_input , _  = next(self.train_iter)
                    placeholders = [self.feat_static_cat , self.past_observed_values
                        , self.past_seasonal_indicators, self.past_time_feat, self.past_target ]
                    feed_dict = dict(zip(placeholders,batch_input))

                    batch_output , _   = sess.run([self.train_result , self.train_op]
                                            , feed_dict=feed_dict)
                    epoch_loss += np.sum(batch_output)
                    avg_epoch_loss = epoch_loss/((batch_no+1)*self.config.batch_size)

                    it.set_postfix(
                        ordered_dict={
                            "avg_epoch_loss":  avg_epoch_loss
                        },
                        refresh=False,
                    )
            toc = time.time()
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
                best_epoch_info['metric_value'],
            )

            if avg_epoch_loss < best_epoch_info['metric_value']:
                best_epoch_info['metric_value'] = avg_epoch_loss
                best_epoch_info['epoch_no'] = epoch_no
                self.save_path = os.path.join(writer_path , "model_params"
                                 ,'best_{}_{}_{}'.format(self.dataset, self.past_length,self.prediction_length)
                            )
                self.saver.save(sess, self.save_path)
                #'/{}_{}_best.ckpt'.format(self.dataset,epoch_no)

        logging.info(
            f"Loading parameters from best epoch "
            f"({best_epoch_info['epoch_no']})"
        )

        logging.info(
            f"Final loss: {best_epoch_info['metric_value']} "
            f"(occurred at epoch {best_epoch_info['epoch_no']})"
        )

        # save net parameters
        logging.getLogger().info("End model training")

    def predict(self):
        sess = self.sess
        self.all_forecast_result = []
        # 如果是 训练之后接着的 predict 直接采用之前 train 产生的save_path
        if hasattr(self ,'save_path'):
            path = self.save_path
            try:
                self.saver.restore(sess, save_path=path)
                print('there is no problem in restoring model params')
            except:
                print('something bad appears')
            finally:
                print('whatever ! life is still fantastic !')

        for batch_no in range(len(self.test_data)//self.batch_size+1):
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
                    print('%d batch %d sample was complicate , throw away' %(batch_no,i))
                    break


        return self.all_forecast_result


    def evaluate(self, forecast=None):
        ground_truth_loader = GroundTruthLoader(config=self.config)
        ground_truth = ground_truth_loader.get_ground_truth()
        if forecast is None:
            forecast = self.all_forecast_result

        for i in range(321):#对于electricity来说，总共由321列
            plot_prob_forecasts(ground_truth[i] , forecast[i] ,i)

        evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
        agg_metrics, item_metrics = evaluator(iter(ground_truth)
                                              , iter(forecast)
                                              , num_series=len(ground_truth_loader.ds.test))

        print(json.dumps(agg_metrics, indent=4))
        pass


def plot_prob_forecasts(ts_entry, forecast_entry, no):
    plot_length = 180
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.savefig('pic/tf_result/result_output_tf_{}.png'.format(no))
    plt.close(fig)