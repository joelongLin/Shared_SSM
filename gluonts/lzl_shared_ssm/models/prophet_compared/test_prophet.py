import pandas as pd
import os
# 这个根据文件系统的目录来决定
# 使用pycharm运行
if '/prophet_compared' in os.getcwd():
    os.chdir('../..')
# 使用bash运行，目录再 pycharm/gluon
elif '/lzl_shared_ssm' not in os.getcwd():
    os.chdir('gluonts/lzl_shared_ssm')
from gluonts.model.prophet import ProphetPredictor
from gluonts.lzl_shared_ssm.utils import create_dataset_if_not_exist ,time_format_from_frequency_str
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

target_data_index = 0
target = 'btc,eth'
environment = 'gold'
maxlags=0
start = '2018-08-02'
freq = '1D'
timestep = 503
pred = 5
plot_length = 5*pred
past = 90
slice = 'overlap'
ds_name_prefix = 'data_process/processed_data/{}_{}_{}_{}.pkl'
save_path = 'logs/btc_eth(prophet)/pred_pic_{}_past({})_pred({})'.format(slice , past , pred)
plot_result_path =  '_'.join([
                                'freq(%s)'%(freq),
                                'lags(%d)'%(maxlags),
                               'past(%d)'%(past)
                                ,'pred(%d)'%(pred)
                             ]
                        )
if not os.path.exists(save_path):
    os.makedirs(save_path)
quantiles = [0.05 , 0.25 , 0.5 , 0.75 , 0.95]
def alpha_for_percentile(p):
    return (p / 100.0) ** 0.3

if __name__ == '__main__':
    ds_name = target.split(',')[target_data_index]
    # 导入 target 以及 environment 的数据
    if slice == 'overlap':
        series = timestep - past - pred + 1
        print('每个数据集的序列数量为 ', series)
    elif slice == 'nolap':
        series = timestep // (past + pred)
        print('每个数据集的序列数量为 ', series)
    else:
        series = 1
        print('每个数据集的序列数量为 ', series ,'情景为长单序列')
    result_root_path = 'evaluate/results'
    forecast_result_saved_path = os.path.join(result_root_path ,'prophet(%s)_'%(target)+plot_result_path+'_2.pkl')
    # 目标序列的数据路径
    target_path = {ds_name: ds_name_prefix.format(
        '%s_start(%s)_freq(%s)' % (ds_name, start,freq), '%s_DsSeries_%d' % (slice, series),
        'train_%d' % past, 'pred_%d' % pred
    ) for ds_name in target.split(',')}

    create_dataset_if_not_exist(
        paths=target_path, start=start, past_length=past
        , pred_length=pred, slice=slice
        , timestep=timestep, freq=freq
    )

    if not os.path.exists(forecast_result_saved_path):
        sample_forecasts = []
        # 由于 target 应该都是 dim = 1 只是确定有多少个 SSM 而已
        for target_name in target_path:
            target = target_path[target_name]
            print('导入原始数据成功~~~')
            with open(target, 'rb') as fp:
                target_ds = pickle.load(fp)
                assert target_ds.metadata['dim'] == 1, 'target 序列的维度都应该为1'
                target_data = target_ds

                prophet_predictor = ProphetPredictor(freq=freq, prediction_length=pred)
                generators = prophet_predictor.predict(target_data.train)
                forecast_samples = list(generators)
                sorted_samples = np.concatenate([np.expand_dims(sample._sorted_samples, 0) for sample in forecast_samples], axis=0)
                sorted_samples = np.expand_dims(sorted_samples, axis=0)
                sample_forecasts.append(sorted_samples)

        sample_forecasts = np.concatenate(sample_forecasts, axis=0)
        with open(forecast_result_saved_path , 'wb') as fp:
            pickle.dump(sample_forecasts , fp)
