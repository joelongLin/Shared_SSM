import tensorflow as tf
import time
import os
import json
# -*-coding:utf-8-*-



def reload_config(FLAGS):

    return FLAGS


def get_image_config():
    cl = tf.app.flags
    relod_model =  'logs/electricity/Nov_11_11:04:04_2019/model_params/best_electricity_336_168'
    # relod_model = ''
    cl.DEFINE_string('reload_model' ,relod_model,'model to reload')
    cl.DEFINE_string('logs_dir','logs/electricity','file to print log')

    # network configuration
    cl.DEFINE_integer('num_layers' ,2,'num of lstm cell layers')
    cl.DEFINE_integer('num_cells' ,40 , 'hidden units size of lstm cell')
    cl.DEFINE_string('cell_type' , 'lstm' , 'Type of recurrent cells to use (available: "lstm" or "gru"')
    cl.DEFINE_float('dropout_rate' , 0.1 , 'Dropout regularization parameter (default: 0.1)')
    cl.DEFINE_string('embedding_dimension' , '' , ' Dimension of the embeddings for categorical features')

    # dataset configuration
    cl.DEFINE_string('dataset' , 'electricity' , 'Name of the target dataset')
    cl.DEFINE_string('freq','1H','Frequency of the data to train on and predict')
    cl.DEFINE_integer('past_length' ,336,'This is the length of the training time series')
    cl.DEFINE_integer('prediction_length' , 168 , 'Length of the prediction horizon')
    cl.DEFINE_bool('add_trend' , False , 'Flag to indicate whether to include trend component in the SSM')

    # prediciton configuration
    cl.DEFINE_integer('num_eval_samples', '100', 'Number of samples paths to draw when computing predictions')
    cl.DEFINE_bool('scaling', True, 'whether to scale the target and observed')
    cl.DEFINE_bool('use_feat_dynamic_real', False, 'Whether to use the ``feat_dynamic_real`` field from the data')
    cl.DEFINE_bool('use_feat_static_cat', True, 'Whether to use the ``feat_static_cat`` field from the data')
    cl.DEFINE_string('cardinality' , '321' , 'Number of values of each categorical feature.')

    #train configuration
    cl.DEFINE_integer('epochs' , 25 , 'Number of epochs that the network will train (default: 1).')
    cl.DEFINE_bool('shuffle' , False ,'whether to shuffle the train dataset')
    cl.DEFINE_integer('batch_size' ,  32 , 'Numbere of examples in each batch')
    cl.DEFINE_integer('num_batches_per_epoch' , 50 , 'Numbers of batches at each epoch')
    cl.DEFINE_float('learning_rate' , 0.001 , 'Initial learning rate')


    # SSM config
    cl.DEFINE_float('noise_emission', 0.03, 'Noise level for the measurement noise matrix')
    cl.DEFINE_float('noise_transition', 0.08, 'Noise level for the process noise matrix')



    return cl


if __name__ == '__main__':
    config = get_image_config()
    config.DEFINE_bool('test', True, 'test')
    config = reload_config(config.FLAGS) #本质 只有 config.FLAGS才是关键

    print(config.dataset)
    config.dataset = 'test'
    print(config.dataset)

    # print(config.__flags)
