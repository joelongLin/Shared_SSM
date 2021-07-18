import pandas as pd
import os
import sys

if '/lzl_shared_ssm' not in os.getcwd():
    os.chdir('gluonts/lzl_shared_ssm')
    sys.path.insert(0, '../..')


from gluonts.model.prophet import ProphetPredictor
from gluonts.lzl_shared_ssm.utils import create_dataset_if_not_exist ,time_format_from_frequency_str, add_time_mark_to_file
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse


parser = argparse.ArgumentParser(description="prophet")
parser.add_argument('-t' ,'--target' , type=str , help='target series', default='btc,eth')
parser.add_argument('-e' ,'--env' , type=str , help='enviroment as dynamic_feat_real', default='gold')
parser.add_argument('-l' , '--lags' , type=int, help='featu_dynamic_real and target series lag', default=5)
parser.add_argument('-st','--start', type=str, help='start time of the dataset',default='2018-08-02')
parser.add_argument('-f'  ,'--freq' , type = str , help = 'frequency of dataset' , default='1D')
parser.add_argument('-T','--timestep', type=int, help='length of the dataset', default=503)
parser.add_argument('-pr'  ,'--pred' , type = int , help = 'length of prediction range' , default=5)
parser.add_argument('-pa'  ,'--past' , type = int , help = 'length of training range' , default=90)
parser.add_argument('-s'  ,'--slice' , type = str , help = 'type of sliding dataset' , default='overlap')
args = parser.parse_args()

target = args.target
start = args.start
# change "_"  in start
if("_" in start):
    start = start.replace("_", " ")
freq = args.freq
timestep = args.timestep
pred = args.pred
past = args.past
slice_style = args.slice
env = args.env
lags = args.lags
ds_name_prefix = 'data_process/processed_data/{}_{}_{}_{}.pkl'

result_params = '_'.join([
                                'freq(%s)'%(freq),
                               'past(%d)'%(past)
                                ,'pred(%d)'%(pred)
                                ,'env(%s)'%(env.replace(',' , '_'))
                                ,'lag(%d)'%(lags)
                             ]
                        )

def add_future_feat_dynamic_real_to_train(target_ds):
    train = target_ds.train.list_data
    test = target_ds.test.list_data
    l = len(train)
    for i in range(l):
        train[i]['feat_dynamic_real'] = test[i]['feat_dynamic_real']

if __name__ == '__main__':
    # load target 
    if slice_style == 'overlap':
        series = timestep - past - pred + 1
        print('Series num in each dataset: ', series)
    elif slice_style == 'nolap':
        series = timestep // (past + pred)
        print('Series num in each dataset: ', series)
    else:
        series = 1
        print('Series num in each dataset: ', series ,', is single series')
    result_root_path  = 'evaluate/results/{}_length({})_slice({})_past({})_pred({})'.format(target.replace(',' ,'_'), timestep , slice_style ,past , pred)
    if not os.path.exists(result_root_path):
        os.makedirs(result_root_path)
    
    forecast_result_saved_path = os.path.join(result_root_path,'PROPHET_'  + result_params + '.pkl')
    forecast_result_saved_path = add_time_mark_to_file(forecast_result_saved_path)

    
    target_path = {ds_name: ds_name_prefix.format(
        '%s_start(%s)_freq(%s)' % (ds_name, start,freq), '%s_DsSeries_%d' % (slice_style, series),
        'train_%d' % past, 'pred_%d_dynamic_feat_real(%s)_lag(%d)' % (pred , env.replace(',', '_'), lags)
    ) for ds_name in target.split(',')}

    for name in target_path:
        path = target_path[name]
        if not os.path.exists(path):
            print(name, "creating ...")
            print(os.getcwd())
            command = "python data_process/preprocessing_env.py --start {} --dataset {} --train_length {} --pred_length {} --slice {} --num_time_steps {} --freq {} --env {} --lags {}".format( args.start, name, past, pred, slice_style, timestep, freq, env, lags)
            print(command)
            os.system(command)
        else:
            print(name , "Dataset exists~~")
    if not os.path.exists(forecast_result_saved_path):
        sample_forecasts = []
        # dimension of each target series should be 1, to check the num of ssm.
        for target_name in target_path:
            target = target_path[target_name]
            print('import data : {}'.format(target))
            with open(target, 'rb') as fp:
                target_ds = pickle.load(fp)
                add_future_feat_dynamic_real_to_train(target_ds)
                
                assert target_ds.metadata['dim'] == 1, 'dimension of target series should be 1'
                
                prophet_predictor = ProphetPredictor(freq=freq, prediction_length=pred)
                generators = prophet_predictor.predict(target_ds.train)
                forecast_samples = list(generators)
                sorted_samples = np.concatenate([np.expand_dims(sample._sorted_samples, 0) for sample in forecast_samples], axis=0)
                sorted_samples = np.expand_dims(sorted_samples, axis=0)
                sample_forecasts.append(sorted_samples)

        sample_forecasts = np.concatenate(sample_forecasts, axis=0)#(ssm_num , bs , num_samples, 1)
        print('Result of prediction are stored in-->', forecast_result_saved_path)
        with open(forecast_result_saved_path , 'wb') as fp:
            pickle.dump(sample_forecasts , fp)
