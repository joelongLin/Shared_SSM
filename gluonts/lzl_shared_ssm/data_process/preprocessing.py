# -*- coding: UTF-8 -*-
# # author : joelonglin
import numpy as np
import pandas as pd
from typing import List, NamedTuple, Optional , Union

import os
if 'lzl_shared_ssm' in os.getcwd():
        if 'data_process' in os.getcwd():
            os.chdir('..')
        # print('------当前与处理数据的路径：', os.getcwd() ,'-------')
else:
    # print('处理数据的时候，请先进入finance的主目录')
    os.chdir('gluonts/lzl_shared_ssm')
# 为了能够让 代码直接使用 ../gluon/gluonts/下的代码
import sys
sys.path.insert(0,
    os.path.abspath(os.path.join(os.getcwd(), "../.."))
)
# print(sys.path)

from gluonts.dataset.common import ListDataset
from gluonts.dataset.common import TrainDatasets
from gluonts.dataset.field_names import FieldName
import pickle


import argparse
from data_info import datasets_info,DatasetInfo

parser = argparse.ArgumentParser(description="data")
parser.add_argument('-st' ,'--start' , type=str , help='start time of dataset', default='2013-04-15 12:00:00')
parser.add_argument('-d','--dataset', type=str, help='name of the needed dataset',default="pressure")
parser.add_argument('-t','--train_length', type=int, help='training length', default=60)
parser.add_argument('-p'  ,'--pred_length' , type = int , help = 'prediction length' , default=5)
parser.add_argument('-s'  ,'--slice' , type = str , help = 'slice style' , default='overlap')
parser.add_argument('-n' , '--num_time_steps' , type=int  , help='timestep of the dataset' , default=1177)
parser.add_argument('-f' , '--freq' , type=str  , help='frequency of the dataset' , default='1H')
args = parser.parse_args()
# change "_"  in start
if("_" in args.start):
    args.start = args.start.replace("_", " ")



# slice the dataset with stride = 1
def slice_df_overlap(
    dataframe ,window_size,past_size
):
    '''
    :param dataframe:  target dataframe
    :param window_size:  windows size
    :param past_size:  training length
    :return:  (1) sliced series (2) start time (3) forecast start time
    '''
    data = dataframe.values
    timestamp = dataframe.index
    target_slice,target_start ,target_forecast_start = [], [] , []
    # feat_static_cat
    for i in range(window_size - 1, len(data)):
        sample = data[(i - window_size + 1):(i + 1)]
        target_slice.append(sample)
        start = timestamp[i-window_size+1]
        target_start.append(start)
        forecast_start = start + past_size*start.freq
        target_forecast_start.append(forecast_start)

    target_slice = np.array(target_slice)
    return target_slice, target_start,target_forecast_start



def slice_df_nolap(
    dataframe,window_size,past_size
):
    data = dataframe.values
    timestamp = dataframe.index
    target_slice, target_start , target_forecast_start= [], [], []
    series = len(data) // window_size
    for j in range(series):
        sample = data[j*window_size : (j+1)*window_size]
        target_slice.append(sample)
        start = timestamp[j*window_size]
        target_start.append(start)
        forecast_start = start + past_size*start.freq
        target_forecast_start.append(forecast_start)
    target_slice = np.array(target_slice)
    return target_slice , target_start , target_forecast_start

# TODO: 这里
slice_func = {
    'overlap' : slice_df_overlap,
    'nolap' : slice_df_nolap
}

def load_finance_from_csv(ds_info):
    '''
    ds_info : DatasetInfo type contain the basic information of raw dataset
    '''
    df = pd.DataFrame()
    path, time_str, aim = ds_info.url, ds_info.time_col, ds_info.aim
    series_start = pd.Timestamp(ts_input=args.start, freq=args.freq)
    series_end = series_start + (args.num_time_steps - 1) * series_start.freq
    # Single file
    if isinstance(path, str):
        url_series = pd.read_csv(path, sep=',', header=0, parse_dates=[time_str])
    # Multiple files, concat on time
    elif isinstance(path, List):
        url_series = pd.DataFrame();
        for file in path:
            file_series = pd.read_csv(file, sep=',', header=0, index_col=ds_info.index_col, parse_dates=[time_str])
            url_series = pd.concat([url_series, file_series], axis=0)

    
    if 'D' in args.freq or 'B' in args.freq:
        url_series[time_str] = pd.to_datetime(url_series[time_str].dt.date)
    else: 
        url_series[time_str] = pd.to_datetime(url_series[time_str])

    url_series.set_index(time_str, inplace=True)
    print("start : " , series_start)
    print("end : " , series_end)
    url_series = url_series.loc[series_start:series_end][aim]

    # 填充额外的数据
    if url_series.shape[0] < args.num_time_steps:
        # padding missing data
        index = pd.date_range(start=series_start, periods=args.num_time_steps, freq=series_start.freq)
        pd_empty = pd.DataFrame(index=index)
        url_series = pd_empty.merge(right=url_series, left_index=True, right_index=True, how='left')
        url_series = url_series.fillna(axis=0, method='ffill')
    elif url_series.shape[0] == args.num_time_steps:
        url_series.set_index(pd.date_range(series_start, series_end, freq=args.freq), inplace=True)
    else:
        print('Invalid Dataset')
        exit()
    
    # Using `url_series[col].index.duplicated()` or `df.index.duplicated()` to check duplicate index problem
    for col in url_series.columns:
        df["{}_{}".format(ds_info.name, col)] = url_series[col]
    return df

def create_dataset(dataset_name):
    ds_info = datasets_info[dataset_name]
    df_aim = load_finance_from_csv(ds_info) #(seq , features)
    ds_metadata = {'prediction_length': args.pred_length,
                   'dim': ds_info.dim,
                   'freq':args.freq,
    }
    func = slice_func.get(args.slice)
    if func != None: #contain slice function
        window_size = args.pred_length + args.train_length
        target_slice , target_start ,target_forecast_start = func(df_aim, window_size,args.train_length)
        ds_metadata['sample_size'] = len(target_slice)
        ds_metadata['num_step'] = window_size
        ds_metadata['start'] = target_start
        ds_metadata['forecast_start'] = target_forecast_start
        return target_slice,ds_metadata
    else: # no slice function
        # feat_static_cat = np.arange(ds_info.dim).astype(np.float32)
        ds_metadata['sample_size'] = 1
        ds_metadata['num_step'] = args.num_time_steps
        time_start = pd.Timestamp(args.start, freq=args.freq)
        ds_metadata['start'] = [time_start
                                for _ in range(datasets_info[dataset_name].dim)]
        ds_metadata['forecast_start'] = [time_start + (args.num_time_steps - args.pred_length) * time_start.freq
                                         for _ in range(datasets_info[dataset_name].dim)]
        return np.expand_dims(df_aim.values, 0) , ds_metadata




def createGluontsDataset(data_name):
    ds_info = datasets_info[data_name]
    # get metadata of target serires
    #(samples_size , seq_len , 1)
    target_slice, ds_metadata = create_dataset(data_name)
    # print('feat_static_cat : ' , feat_static_cat , '  type:' ,type(feat_static_cat))
    train_ds = ListDataset([{FieldName.TARGET: target,
                             FieldName.START: start,
                             FieldName.FORECAST_START: forecast_start}
                            for (target, start, forecast_start) in
                            zip(target_slice[:, :-ds_metadata['prediction_length']],
                                ds_metadata['start'], ds_metadata['forecast_start']
                                )],
                           freq=ds_metadata['freq'],
                           one_dim_target=False)

    test_ds = ListDataset([{FieldName.TARGET: target,
                            FieldName.START: start,
                            FieldName.FORECAST_START: forecast_start}
                           for (target, start, forecast_start) in zip(target_slice,
                                                                      ds_metadata['start'],
                                                                      ds_metadata['forecast_start']
                                                                      )],
                          freq=ds_metadata['freq'],
                          one_dim_target=False)

    dataset = TrainDatasets(metadata=ds_metadata, train=train_ds, test=test_ds)
    with open('data_process/processed_data/{}_{}_{}_{}.pkl'.format(
            '%s_start(%s)_freq(%s)'%(data_name, args.start,args.freq), '%s_DsSeries_%d'%(args.slice,dataset.metadata['sample_size']),
            'train_%d'%args.train_length, 'pred_%d'%args.pred_length,
    ), 'wb') as fp:
        pickle.dump(dataset, fp)

    print('current dataset: ', data_name , 'training length %d ,prediction length %d '%(args.train_length , args.pred_length),
          'sample number of dataset: ：'  , dataset.metadata['sample_size'])

if __name__ == '__main__':
          
        

    finance_data_name = args.dataset
    createGluontsDataset(finance_data_name)



