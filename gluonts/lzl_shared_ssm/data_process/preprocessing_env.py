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

# cryptcurrency : btc eth 503 gold 2018-08-02
# economic indices : UKX VIX SPX SHSZ300 NKY 2606 SX5E,DXY '2008-01-04'
parser.add_argument('-st' ,'--start' , type=str , help='数据集开始的时间', default='2018-08-02')
parser.add_argument('-d','--dataset', type=str, help='需要重新生成的数据的名称',default='btc')
parser.add_argument('-e' , '--env' , type=str, help='赋值给 feat_dynamic_real 的特征', default='gold')
parser.add_argument('-l' , '--lags' , type=int, help='featu_dynamic_real和目标序列之间的lag', default=5)
parser.add_argument('-t','--train_length', type=int, help='数据集训练长度', default=90)
parser.add_argument('-p'  ,'--pred_length' , type = int , help = '需要预测的长度' , default=5)
parser.add_argument('-s'  ,'--slice' , type = str , help = '需要预测的长度' , default='overlap')
parser.add_argument('-n' , '--num_time_steps' , type=int  , help='时间步的数量' , default=503)
parser.add_argument('-f' , '--freq' , type=str  , help='时间间隔' , default='1D')
args = parser.parse_args()
# change "_"  in start
if("_" in args.start):
    args.start = args.start.replace("_", " ")

# 切割完之后， 除了目标序列target_slice 之外
# 此方法用于 stride = 1, 完全滑动窗口
def slice_df_overlap(
    dataframe ,window_size,past_size
):
    '''
    :param dataframe:  给定需要滑动切片的总数据集
    :param window_size:  窗口的大小
    :param past_size:  训练的窗口大小
    :return:  (2) 切片后的序列 (2)序列开始的时间 List[start_time] (3) 序列开始预测的时间
    '''
    data = dataframe.values
    timestamp = dataframe.index
    target_slice,target_start ,target_forecast_start = [], [] , []
    # feat_static_cat 特征值，还是按照原序列进行标注
    for i in range(window_size - 1, len(data)):
        sample = data[(i - window_size + 1):(i + 1)]
        target_slice.append(sample)
        start = timestamp[i-window_size+1]
        target_start.append(start)
        forecast_start = start + past_size*start.freq
        target_forecast_start.append(forecast_start)

    target_slice = np.array(target_slice)
    return target_slice, target_start,target_forecast_start

# 此方法对序列的切割，完全不存在重叠
# 缺点：如果窗口和长度不存在倍数关系，会使得数据集存在缺失
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

# TODO: 这里选择不同的方式
slice_func = {
    'overlap' : slice_df_overlap,
    'nolap' : slice_df_nolap
}
#这里特别容易出现问题
def load_finance_from_csv(ds_info, flag):
    '''
        @ ds_info : DatasetInfo 类型，包含序列的信息
        @ flag : 当前需要处理的数据集到底是 "env"(外部信息)还是"target"(目标序列)
    '''
    df = pd.DataFrame()
    path, time_str, aim = ds_info.url, ds_info.time_col, ds_info.aim

    target_start = pd.Timestamp(args.start,freq=args.freq)
    # 如果当前的数据是作为背景信息
    if flag == "env":
        series_start = target_start - args.lags * target_start.freq
    else:
        series_start = target_start
    # series_start = pd.Timestamp(ts_input=args.start, freq=args.freq)

    series_end = series_start + (args.num_time_steps - 1) * series_start.freq
    # 单文件数据集
    if isinstance(path, str):
        url_series = pd.read_csv(path, sep=',', header=0, parse_dates=[time_str])
    # 多文件数据集 ，进行 时间轴 上的拼接
    elif isinstance(path, List):
        url_series = pd.DataFrame();
        for file in path:
            file_series = pd.read_csv(file, sep=',', header=0, index_col=ds_info.index_col, parse_dates=[time_str])
            url_series = pd.concat([url_series, file_series], axis=0)

    # TODO: 这里要注意不够 general 只在适合 freq='D' 或者 freq = 'B' 的情况
    if 'D' in args.freq or 'B' in args.freq:
        url_series[time_str] = pd.to_datetime(url_series[time_str].dt.date)

    url_series.set_index(time_str, inplace=True)
    url_series = url_series.loc[series_start:series_end][aim]

    # NOTE: statistic the misssing date
    expected_date = pd.date_range(start=series_start, periods=args.num_time_steps, freq=series_start.freq)
    missing_date = []
    for i in expected_date:
        if i not in url_series.index:
            missing_date.append(i)
    print("==========missing day num:", len(missing_date) ,"======")
    import holidays
    UK_holidays = holidays.UK()
    US_holidays = holidays.US()
    for day in missing_date:
        print('missing : ', day , ' dayofweek' , day.isoweekday() ,' holidayofUS' , US_holidays.get(day))
    

    if url_series.shape[0] < args.num_time_steps:
        # 添加缺失的时刻(比如周末数据缺失这样的)
        index = pd.date_range(start=series_start, periods=args.num_time_steps, freq=series_start.freq)
        pd_empty = pd.DataFrame(index=index)
        url_series = pd_empty.merge(right=url_series, left_index=True, right_index=True, how='left')
        url_series = url_series.fillna(axis=0, method='ffill')
    elif url_series.shape[0] == args.num_time_steps:
        url_series.set_index(pd.date_range(series_start, series_end, freq=args.freq), inplace=True)
    else:
        print('数据集有问题，请重新检查')
        exit()
    # TODO: 这里因为当第二个 pd.Series要插入到 df 中出现错误时，切记要从初始数据集的脏数据出发
    # 使用 url_series[col].index.duplicated()  或者  df.index.duplicated() 查看是否存在index重复的问题
    for col in url_series.columns:
        df["{}_{}".format(ds_info.name, col)] = url_series[col]
    return df

def create_dataset(dataset_name , flag):
    '''
    @ dataset_name : 需要处理的数据集的名称，产生所有的数据集
    @ flag : 当前需要处理的数据集到底是 "env"(外部信息)还是"target"(目标序列)
    '''
    ds_info = datasets_info[dataset_name]
    df_aim = load_finance_from_csv(ds_info, flag) #(seq , features)
    ds_metadata = {'prediction_length': args.pred_length,
                   'dim': ds_info.dim,
                   'freq':args.freq,
    }
    func = slice_func.get(args.slice)

    target_start = pd.Timestamp(args.start,freq=args.freq)
    # 如果当前的数据是作为背景信息
    if flag == "env":
        time_start = target_start - args.lags * target_start.freq
    else:
        time_start = target_start

    if func != None: #表示存在窗口切割函数
        window_size = args.pred_length + args.train_length
        target_slice , target_start ,target_forecast_start = func(df_aim, window_size,args.train_length)
        ds_metadata['sample_size'] = len(target_slice)
        ds_metadata['num_step'] = window_size
        ds_metadata['start'] = target_start
        ds_metadata['forecast_start'] = target_forecast_start
        return target_slice,ds_metadata
    else: # 表示该数据集不进行切割
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
    @ data_name : 需要处理的数据集
    @ flag : 当前需要处理的数据集到底是 "env"(外部信息)还是"target"(目标序列)
    '''
    ds_info = datasets_info[data_name]
    # 获取所有目标序列，元信息
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
    print('当前数据集为: ', data_name , '训练长度为 %d , 预测长度为 %d '%(args.train_length , args.pred_length),
          '(切片之后)每个数据集样本数目为：'  , dataset.metadata['sample_size'])
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
    # 直接在外围进行拼凑
    for env_ds in env_gluon_dataset:
        # 在训练集中添加
        for i in range(target_ds.metadata['sample_size']):
            train_env_cell = env_ds.train.list_data[i]
            train_target_cell = target_ds.train.list_data[i];
            test_env_cell = env_ds.test.list_data[i]
            test_target_cell = target_ds.test.list_data[i];
            
            #! 这里的FEAT_DYNAMIC_REAL 必须采用 (C, length)的方式
            #! 同时有必要进行归一化
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
        # 在测试集合中添加

    # 最后进行存储
    with open('data_process/processed_data/{}_{}_{}_{}.pkl'.format(
            '%s_start(%s)_freq(%s)'%(finance_data_name, args.start,args.freq), '%s_DsSeries_%d'%(args.slice,target_ds.metadata['sample_size']),
            'train_%d'%args.train_length, 'pred_%d_dynamic_feat_real(%s)_lag(%d)'%(args.pred_length, args.env.replace(',' , '_'), args.lags)
    ), 'wb') as fp:
        pickle.dump(target_ds, fp)