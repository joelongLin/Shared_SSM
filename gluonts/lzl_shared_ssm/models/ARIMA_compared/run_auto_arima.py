# -*- coding: UTF-8 -*-
# author : joelonglin
import pandas as pd
import os
import sys
# 使用bash运行，目录再 ..../gluon
if '/lzl_shared_ssm' not in os.getcwd():
    os.chdir('gluonts/lzl_shared_ssm')
    sys.path.insert(0, '../..')

import pandas as pd
from gluonts.lzl_shared_ssm.utils import create_dataset_if_not_exist , add_time_mark_to_file
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pmdarima as pm
from pmdarima.model_selection import train_test_split
from pmdarima.pipeline import Pipeline
from pmdarima.preprocessing import BoxCoxEndogTransformer

parser = argparse.ArgumentParser(description="arima")
parser.add_argument('-t' ,'--target' , type=str , help='target series', default='btc,eth')
parser.add_argument('-st','--start', type=str, help='start time of the dataset',default='2018-08-02')
parser.add_argument('-f'  ,'--freq' , type = str , help = 'frequency of dataset' , default='1D')
parser.add_argument('-T','--timestep', type=int, help='length of the dataset', default=503)
parser.add_argument('-pr'  ,'--pred' , type = int , help = 'length of prediction range' , default=1)
parser.add_argument('-pa'  ,'--past' , type = int , help = 'length of training range' , default=30)
parser.add_argument('-s'  ,'--slice' , type = str , help = 'type of sliding dataset' , default='overlap')
args = parser.parse_args()

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
    forecast_result_saved_path = os.path.join(result_root_path,'arima(%s)_' % (target.replace(',' ,'_')) + result_params + '.pkl')
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
        forecasts = []

        for target_name in target_path:
            target = target_path[target_name]
            print('导入数据 : {}'.format(target))
            with open(target, 'rb') as fp:
                target_ds = pickle.load(fp)
                assert target_ds.metadata['dim'] == 1, 'target 序列的维度都应该为1'
                # for sample in 
                ds_forecast = []
                for entry_no,  train_entry in enumerate(target_ds.train.list_data):
                    #构造一个pd Series
                    train = pd.Series(
                        data=train_entry['target'].squeeze(-1),
                        index=pd.date_range(start=train_entry['start'] ,periods=train_entry['target'].shape[0] , freq=freq) 
                    )
                    
                    # print(np.where(train.isna() == True))
                    
                    pipeline = Pipeline([  
                        ('boxcox', BoxCoxEndogTransformer(lmbda=1.0,lmbda2=1e-3 ,neg_action="ignore")),  # lmbda2 avoids negative values
                        ('arima', pm.AutoARIMA(seasonal=False, m=1, 
                                            suppress_warnings=True,
                                            trace=True,error_action="ignore"))
                    ])

                    print('当前的是{}数据的第{}个entry,其起始时间为{},其取值为na的下标为'.format(
                        target_name ,entry_no , train_entry['start'])
                    )

                    # if entry_no == 378 and target_name == 'eth':
                    pipeline.fit(train )
                    
                    pred_result = pipeline.predict(pred)
                    ds_forecast.append(np.expand_dims(pred_result,axis=0))
                ds_forecast = np.concatenate(ds_forecast , axis=0) #(bs, pred)
            
            forecasts.append(np.expand_dims(ds_forecast, axis=0))
        
        forecasts = np.concatenate(forecasts , axis=0) #(ssm_num , bs, pred)
        print('把预测结果保存在-->', forecast_result_saved_path)
        with open(forecast_result_saved_path , 'wb') as fp:
            pickle.dump(forecasts , fp)

                
    
               
                
               
               
            
    
            


    