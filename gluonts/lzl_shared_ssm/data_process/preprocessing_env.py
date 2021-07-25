# -*- coding: UTF-8 -*-
# # author : joelonglin
import numpy as np
import pandas as pd
from typing import List, NamedTuple, Optional , Union
import os
if 'lzl_shared_ssm' in os.getcwd():
        if 'data_process' in os.getcwd():
            os.chdir('..')
else:
    os.chdir('gluonts/lzl_shared_ssm')

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

# cryptcurrency : btc eth 503 gold 2018-08-02
# economic indices : UKX VIX SPX SHSZ300 NKY 2606 SX5E,DXY '2008-01-04'
parser.add_argument('-st' ,'--start' , type=str , help='start time of dataset', default='2018-08-02')
parser.add_argument('-d','--dataset', type=str, help='name of the needed dataset',default='btc')
parser.add_argument('-e' , '--env' , type=str, help=' feat_dynamic_real ', default='gold')
parser.add_argument('-l' , '--lags' , type=int, help='featu_dynamic_real and target series lag', default=5)
parser.add_argument('-t','--train_length', type=int, help='training length', default=90)
parser.add_argument('-p'  ,'--pred_length' , type = int , help = 'prediction length' , default=5)
parser.add_argument('-s'  ,'--slice' , type = str , help = 'prediction length' , default='overlap')
parser.add_argument('-n' , '--num_time_steps' , type=int  , help='timestep of the dataset' , default=503)
parser.add_argument('-f' , '--freq' , type=str  , help='frequency of the dataset' , default='1D')
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


slice_func = {
    'overlap' : slice_df_overlap,
    'nolap' : slice_df_nolap
}

def load_finance_from_csv(ds_info, flag):
    '''
        @ ds_info : Seires information
        @ flag : ENV or TARGET
    '''
    df = pd.DataFrame()
    path, time_str, aim = ds_info.url, ds_info.time_col, ds_info.aim

    target_start = pd.Timestamp(args.start,freq=args.freq)
    
    if flag == "env":
        series_start = target_start - args.lags * target_start.freq
    else:
        series_start = target_start
    # series_start = pd.Timestamp(ts_input=args.start, freq=args.freq)

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
    url_series = url_series.loc[series_start:series_end][aim]
    

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

def create_dataset(dataset_name , flag):
    '''
    @ dataset_name : needed dataset
    @ flag : ENV or TARGET
    '''
    ds_info = datasets_info[dataset_name]
    df_aim = load_finance_from_csv(ds_info, flag) #(seq , features)
    ds_metadata = {'prediction_length': args.pred_length,
                   'dim': ds_info.dim,
                   'freq':args.freq,
    }
    func = slice_func.get(args.slice)

    target_start = pd.Timestamp(args.start,freq=args.freq)
    
    if flag == "env":
        time_start = target_start - args.lags * target_start.freq
    else:
        time_start = target_start

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
        ds_metadata['start'] = [time_start
                                for _ in range(datasets_info[dataset_name].dim)]
        ds_metadata['forecast_start'] = [time_start + (args.num_time_steps - args.pred_length) * time_start.freq
                                         for _ in range(datasets_info[dataset_name].dim)]
        return np.expand_dims(df_aim.values, 0) , ds_metadata




def createGluontsDataset(data_name:str , flag :str):
    '''
    @ data_name : needed dataset
    @ flag : ENV or TARGET
    '''
    ds_info = datasets_info[data_name]
    # get metadata of target serires
    #(samples_size , seq_len , 1)
    target_slice, ds_metadata = create_dataset(data_name , flag)
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
    print('current dataset: ', data_name , 'training length %d ,prediction length %d '%(args.train_length , args.pred_length),
          'sample number of dataset: ：'  , dataset.metadata['sample_size'])
    return dataset;

if __name__ == '__main__':
    FEAT_DYNAMIC_REAL = "feat_dynamic_real"
    finance_data_name = args.dataset
    envs = args.env.split(',')
    env_gluon_dataset = []
    for env_name in envs:
        env_ds = createGluontsDataset(env_name, "env")
        env_gluon_dataset.append (env_ds)
    target_ds = createGluontsDataset(finance_data_name, "target")

    #'prediction_length': 5, 'dim': 1, 'freq': '1D', 'sample_size': 409,
    
    for env_ds in env_gluon_dataset:
        
        for i in range(target_ds.metadata['sample_size']):
            train_env_cell = env_ds.train.list_data[i]
            train_target_cell = target_ds.train.list_data[i];
            test_env_cell = env_ds.test.list_data[i]
            test_target_cell = target_ds.test.list_data[i];
            
            #! FEAT_DYNAMIC_REAL's dimension should be (C, length)
            #! Normalization
            T = np.transpose(train_env_cell["target"])
            normalized = np.divide(T, np.expand_dims(np.nanmean(T, 1), 1))
            normalized[np.isnan(normalized)] = 0;
            if FEAT_DYNAMIC_REAL not in train_target_cell:
                train_target_cell[FEAT_DYNAMIC_REAL] = normalized
            else:
                train_target_cell[FEAT_DYNAMIC_REAL] = np.concatenate([train_target_cell[FEAT_DYNAMIC_REAL] ,normalized] , axis=0);
            
            T = np.transpose(test_env_cell["target"])
            normalized = np.divide(T, np.expand_dims(np.nanmean(T[:,:args.train_length], 1), 1))
            normalized[np.isnan(normalized)] = 0;
            if FEAT_DYNAMIC_REAL not in test_target_cell:
                test_target_cell[FEAT_DYNAMIC_REAL] = normalized
            else:
                test_target_cell[FEAT_DYNAMIC_REAL] = np.concatenate([test_target_cell[FEAT_DYNAMIC_REAL] , normalized] , axis=0);

    # store
    with open('data_process/processed_data/{}_{}_{}_{}.pkl'.format(
            '%s_start(%s)_freq(%s)'%(finance_data_name, args.start,args.freq), '%s_DsSeries_%d'%(args.slice,target_ds.metadata['sample_size']),
            'train_%d'%args.train_length, 'pred_%d_dynamic_feat_real(%s)_lag(%d)'%(args.pred_length, args.env.replace(',' , '_'), args.lags)
    ), 'wb') as fp:
        pickle.dump(target_ds, fp)