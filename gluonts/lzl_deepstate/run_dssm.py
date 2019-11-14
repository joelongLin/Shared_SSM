# -*- utf-8 -*-
# author : joelonglin

from gluonts.lzl_deepstate.model.deepstate import DeepStateNetwork
import pickle
from gluonts.lzl_deepstate.utils.config import get_image_config , reload_config
from gluonts.lzl_deepstate.model.issm import LevelISSM, LevelTrendISSM,SeasonalityISSM
import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# argparser
config = get_image_config()
config = reload_config(config.FLAGS)

if ('/lzl_deepstate' not in os.getcwd()):
     os.chdir('gluonts/lzl_deepstate')
     print('change os dir : ',os.getcwd())

configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
with tf.compat.v1.Session(config=configuration) as sess:
    dssm = DeepStateNetwork(config=config,sess=sess)\
        .build_module().build_train_forward().build_predict_forward().initialize_variables()
    dssm.train()
    dssm.predict()
    dssm.evaluate()



# ssm = SeasonalityISSM(7)
# seasonal_indicators = tf.random.normal(shape=[16,48,2])
# seasonal_indicators = tf.expand_dims(tf.random.categorical(tf.random.normal([16,7]), 48),axis=-1)
# season_1 = tf.expand_dims(tf.random.categorical(tf.random.normal([16,24]), 48),axis=-1)
# season_2 = tf.expand_dims(tf.random.categorical(tf.random.normal([16,7]), 48),axis=-1)
# season = tf.concat([season_1,season_2],axis=-1)
# print(seasonal_indicators)
# coef_emiss = ssm.emission_coeff(seasonal_indicators)
# coef_innov = ssm.innovation_coeff(seasonal_indicators)
# coef_trans = ssm.transition_coeff(seasonal_indicators)
# print(coef_emiss,'\n',coef_innov,'\n',coef_trans)

# features_1 = tf.expand_dims(tf.random.categorical(tf.random.normal([16,2]), 48),axis=-1)
# features_2 = tf.expand_dims(tf.random.categorical(tf.random.normal([16,3]), 48),axis=-1)
# features = tf.concat([features_1,features_2],axis=-1)

