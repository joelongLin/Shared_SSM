# -*- coding: UTF-8 -*-
# author : joelonglin
import os
import sys
sys.path.insert(0,os.getcwd())
if ('/lzl_shared_ssm' not in os.getcwd()):
    os.chdir('gluonts/lzl_shared_ssm')
    print('change os dir : ', os.getcwd())
from gluonts.lzl_shared_ssm.models.shared_SSM import SharedSSM
import pickle
from gluonts.lzl_shared_ssm.utils import get_reload_hyper
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



cl = tf.app.flags
reload_model = 'freq(1D)_env(gold)_lags(5)_past(90)_pred(5)_u(5)_l(4)_K(2)_T(2-40)_E(2-50)_α(50)_epoch(100)_bs(32)_bn(13)_lr(0.001)_initKF(0.05)_dropout(0.5)'
cl.DEFINE_string('logs_dir','logs/btc_eth(shared_ssm)','file to print log')
cl.DEFINE_string('reload_model' ,reload_model,'models to reload')
cl.DEFINE_string('reload_time' , '' , 'time marker of the reload model')


# dataset configuration
cl.DEFINE_bool('scaling' , True , 'whether use Mean Scaler to preprocess the data')
cl.DEFINE_string('target' , 'btc,eth' , 'Name of the target dataset')
cl.DEFINE_string('environment' , 'gold' , 'Name of the dataset ')
cl.DEFINE_string('start' , '2018-08-02' ,'start of the training range')
cl.DEFINE_integer('timestep' , 503 , 'length of the series') #这个序列的长度实际上也决定了样本数量的大小
cl.DEFINE_string('slice' , 'overlap' , 'how to slice the dataset')
cl.DEFINE_string('freq','1D','Frequency of the data to train on and predict')
cl.DEFINE_integer('past_length' ,90,'This is the length of the training time series')
cl.DEFINE_integer('pred_length' , 5 , 'Length of the prediction horizon')

#train configuration
cl.DEFINE_integer('epochs' , 100 , 'Number of epochs that the network will train (default: 1).')
cl.DEFINE_bool('shuffle' , False ,'whether to shuffle the train dataset')
cl.DEFINE_integer('batch_size' ,  10 , 'Numbere of examples in each batch')
cl.DEFINE_integer('num_batches_per_epoch' , 2 , 'Numbers of batches at each epoch')
cl.DEFINE_string('gpu' , '3' , 'gpu to use')
cl.DEFINE_float('dropout_rate' , 0.5 , 'Dropout regularization parameter (default: 0.1)')
cl.DEFINE_float('learning_rate' , 0.001 , 'Initial learning rate')


# SSM configuration
cl.DEFINE_integer('dim_z', 1, 'Dimension of the observation in the LGSSM')
cl.DEFINE_integer('dim_l', 4, 'Dimension of the latent state in the LGSSM')
# TODO: 这里面的 dim_u 由 LSTM中提取公共信息特征维度，最好能够有一个可解释性的维度
cl.DEFINE_integer('dim_u', 5 , 'Dimension of the inputs , which come from LSTM')
cl.DEFINE_integer('K', 2, 'Number of filters in mixture')
# TODO：这里关于Noise的强度，也可以参考DeepState中，根据timestamp生成的特征获取噪声的大小
# cl.DEFINE_float('noise_emission', 0.03, 'Noise level for the measurement noise matrix')
# cl.DEFINE_float('noise_transition', 0.08, 'Noise level for the process noise matrix')
cl.DEFINE_float('init_kf_matrices', 0.05, 'initialize the B and C matrix')
cl.DEFINE_float('init_cov', 10.0, 'Variance of the initial state')

# shared time_feature network configuration
cl.DEFINE_integer('time_exact_layers' ,2,'num of lstm cell layers')
cl.DEFINE_integer('time_exact_cells' ,40 , 'hidden units size of lstm cell')
cl.DEFINE_string('time_cell_type' , 'lstm' , 'Type of recurrent cells to use (available: "lstm" or "gru"')

# shared environment network configuration
cl.DEFINE_bool('use_env' , True , 'whether use environment variable to generate control seq' )
cl.DEFINE_integer('maxlags' , 5 , 'the time lag between environment variable and target variable')
cl.DEFINE_integer('env_exact_layers' ,2,'num of lstm cell layers')
cl.DEFINE_integer('env_exact_cells' ,50 , 'hidden units size of lstm cell')
cl.DEFINE_string('env_cell_type' , 'lstm' , 'Type of recurrent cells to use (available: "lstm" or "gru"')

# alpha network configuration
cl.DEFINE_boolean('alpha_rnn', True, 'Use LSTM RNN for alpha')
cl.DEFINE_integer('alpha_units', 50, 'Number of units in alpha network')


# prediciton configuration
cl.DEFINE_integer('num_samples', '1', 'Number of samples paths to draw when computing predictions')




def main(_):
    config = cl.FLAGS
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    if config.reload_model != '':
        print('reload model : ' , config.reload_model)
        config = get_reload_hyper(config.reload_model , config)

    # config = reload_config(config)
    configuration = tf.compat.v1.ConfigProto()
    configuration.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=configuration) as sess:
        sharedSSM = SharedSSM(config=config, sess=sess)\
            .build_module().build_train_forward().build_predict_forward().initialize_variables()
        sharedSSM.train()
        sharedSSM.predict()
        # sharedSSM.evaluate()



if __name__ == '__main__':
   tf.app.run()


# for i in config:
#     try:
#         print(i , ':' , eval('config.{}'.format(i)))
#     except:
#         print('当前 ' , i ,' 属性获取有问题')
#         continue
# exit()