import pandas as pd
import os
import sys
# 使用bash运行，目录再 pycharm/gluon
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
parser.add_argument('-st','--start', type=str, help='start time of the dataset',default='2018-08-02')
parser.add_argument('-f'  ,'--freq' , type = str , help = 'frequency of dataset' , default='1D')
parser.add_argument('-T','--timestep', type=int, help='length of the dataset', default=637)
parser.add_argument('-pr'  ,'--pred' , type = int , help = 'length of prediction range' , default=5)
parser.add_argument('-pa'  ,'--past' , type = int , help = 'length of training range' , default=30)
parser.add_argument('-s'  ,'--slice' , type = str , help = 'type of sliding dataset' , default='nolap')
args = parser.parse_args()

# change "_"  in start
if("_" in args.start):
    args.start = args.start.replace("_", " ")

target = args.target
start = args.start
freq = args.freq
timestep = args.timestep
pred = args.pred
past = args.past
slice_style = args.slice
ds_name_prefix = 'data_process/processed_data/{}_{}_{}_{}.pkl'

result_params = '_'.join([
                                'freq(%s)'%(freq),
                               'past(%d)'%(past)
                                ,'pred(%d)'%(pred)
                             ]
                        )


if __name__ == '__main__':
    # 导入 target 
    if slice_style == 'overlap':
        series = timestep - past - pred + 1
        print('每个数据集的序列数量为 ', series)
    elif slice_style == 'nolap':
        series = timestep // (past + pred)
        print('每个数据集的序列数量为 ', series)
    else:
        series = 1
        print('每个数据集的序列数量为 ', series ,'情景为长单序列')
    result_root_path  = 'evaluate/results/{}_length({})_slice({})_past({})_pred({})'.format(target.replace(',' ,'_'), timestep , slice_style ,past , pred)
    if not os.path.exists(result_root_path):
        os.makedirs(result_root_path)
    forecast_result_saved_path = os.path.join(result_root_path,'prophet_'  + result_params + '.pkl')
    forecast_result_saved_path = add_time_mark_to_file(forecast_result_saved_path)

    # 目标序列的数据路径
    target_path = {ds_name: ds_name_prefix.format(
        '%s_start(%s)_freq(%s)' % (ds_name, start,freq), '%s_DsSeries_%d' % (slice_style, series),
        'train_%d' % past, 'pred_%d' % pred
    ) for ds_name in target.split(',')}

    create_dataset_if_not_exist(
        paths=target_path, start=start, past_length=past
        , pred_length=pred, slice=slice_style
        , timestep=timestep, freq=freq
    )

    if not os.path.exists(forecast_result_saved_path):
        sample_forecasts = []
        # 由于 target 应该都是 dim = 1 只是确定有多少个 SSM 而已
        for target_name in target_path:
            target = target_path[target_name]
            print('导入数据 : {}'.format(target))
            with open(target, 'rb') as fp:
                target_ds = pickle.load(fp)
                assert target_ds.metadata['dim'] == 1, 'target 序列的维度都应该为1'
                
                prophet_predictor = ProphetPredictor(freq=freq, prediction_length=pred)
                generators = prophet_predictor.predict(target_ds.train)
                forecast_samples = list(generators)
                sorted_samples = np.concatenate([np.expand_dims(sample._sorted_samples, 0) for sample in forecast_samples], axis=0)
                sorted_samples = np.expand_dims(sorted_samples, axis=0)
                sample_forecasts.append(sorted_samples)

        sample_forecasts = np.concatenate(sample_forecasts, axis=0)#(ssm_num , bs , num_samples, 1)
        print('把预测结果保存在-->', forecast_result_saved_path)
        with open(forecast_result_saved_path , 'wb') as fp:
            pickle.dump(sample_forecasts , fp)
