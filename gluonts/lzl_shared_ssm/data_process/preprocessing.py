# -*- coding: UTF-8 -*-
# author : joelonglin
import numpy as np
import pandas as pd
from typing import List, NamedTuple, Optional ,Union
from gluonts.dataset.common import ListDataset
from gluonts.dataset.common import TrainDatasets
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import to_pandas
import matplotlib.pyplot as plt
import pickle
import os
import argparse


parser = argparse.ArgumentParser(description="data")
parser.add_argument('-d','--dataset', type=str, help='需要重新生成的数据的名称',default='eth')
parser.add_argument('-t','--train_length', type=int, help='数据集训练长度', default=30)
parser.add_argument('-p'  ,'--pred_length' , type = int , help = '需要预测的长度' , default=1)
parser.add_argument('-s'  ,'--slice' , type = str , help = '需要预测的长度' , default='overlap')
parser.add_argument('-n' , '--num_time_steps' , type=int  , help='时间步的数量' , default='500')
parser.add_argument('-f' , '--freq' , type=str  , help='时间间隔' , default='1D')
args = parser.parse_args()

root='data_process/raw_data/'
class DatasetInfo(NamedTuple):
    name: str  # 该数据集的名称
    url: str  #存放该数据集的
    timestamp : str  #表明这些 url 里面对应的 表示时刻的名称 如: eth.csv 中的 beijing_time  或者 btc.csv 中的Date
    dim: int  #该数据集存在多少条序列
    num_time_steps: int # 该数据集对应的序列的数量
    train_length:int    #训练长度
    prediction_length: int  #预测长度
    slice:str  #是否进行切片('no' ,'overover' ,'nolap')，切片的长度是train_length + prediction_length
    freq: str   # 该序列时间步的大小
    start_date: str #序列开始的时间
    aim : List[str]  # 序列的目标特征，包含的数量特征应该与 url， timestamp一致
    feat_dynamic_cat: Optional[str] = None  #表明该序列类别的 (seq_length , category)

datasets_info = {
    "btc": DatasetInfo(
        name="btc",
        url=root + 'btc.csv',
        timestamp='Date',
        dim=1,
        num_time_steps=args.num_time_steps,
        train_length=args.train_length,
        prediction_length=args.pred_length,
        slice=args.slice,
        start_date="2018-08-02",
        aim=['close'],
        freq=args.freq,
    ),
    "eth": DatasetInfo(
        name="eth",
        url=root + 'eth.csv',
        timestamp='beijing_time',
        dim=1,
        num_time_steps=args.num_time_steps,
        train_length=args.train_length,
        prediction_length=args.pred_length,
        slice=args.slice,
        start_date="2018-08-02",
        aim=['close'],  # 这样子写只是为了测试预处理程序是否强大
        freq=args.freq,
    ),
    "gold": DatasetInfo(
        name="gold",
        url=root + 'GOLD.csv',
        timestamp='Date',
        dim=2,
        num_time_steps=503,
        train_length=args.train_length,
        prediction_length=args.pred_length,
        slice=args.slice,
        start_date="2018-08-02",
        aim=['Open','Close'],
        freq=args.freq,
    ),
}
# 切割完之后， 除了目标序列target_slice 之外
# TODO: 我去掉了DeepState 里面的 feat_static_cat, 因为真的不用给样本进行编号
def slice_df_overlap(
    dataframe:pd.DataFrame
    ,window_size:int
):
    data = dataframe.values
    timestamp = dataframe.index
    # TODO: 思考一下 DeepState中的 feat_static_cat 有什么用？
    target_slice,target_start = [], []
    _ , dim = data.shape
    # feat_static_cat 特征值，还是按照原序列进行标注
    for i in range(window_size - 1, len(data)):
        a = data[(i - window_size + 1):(i + 1)]
        target_slice.append(a)
        target_start.append(timestamp[i-window_size+1])

    target_slice = np.array(target_slice)
    return target_slice, target_start

# TODO: 暂时没空写这个方法，后面有空写上
def slice_df_nolap(
    dataframe:pd.DataFrame
    ,window_size:int
):
    data = dataframe.values
    timestamp = dataframe.index
    target_slice, target_start, feat_static_cat, label = [], [], [], []
    _, dim = data.shape
    for j in range(dim):
        # 这里面 根据窗口长度选择长度
        for i in range((len(data) // window_size)+1):
            pass
    pass

slice_func = {
    'overlap' : slice_df_overlap,
    'nolap' : slice_df_nolap
}
def load_finance_from_csv(ds_info):
    df = pd.DataFrame()
    path, time_str,aim  = ds_info.url, ds_info.timestamp , ds_info.aim
    series_start = pd.Timestamp(ts_input=ds_info.start_date , freq=ds_info.freq)
    series_end = series_start + (ds_info.num_time_steps-1)*series_start.freq
    url_series = pd.read_csv(path ,sep=',' ,header=0 , parse_dates=[time_str])
    # TODO: 这里要注意不够 general 只在适合 freq='D' 的情况
    if 'D' in ds_info.freq:
        url_series[time_str] = pd.to_datetime(url_series[time_str].dt.date)
    url_series.set_index(time_str ,inplace=True)
    url_series = url_series.loc[series_start:series_end][aim]
    if url_series.shape[0] < ds_info.num_time_steps:
        #添加缺失的时刻(比如周末数据确实这样的)
        index = pd.date_range(start=series_start, periods=ds_info.num_time_steps, freq=series_start.freq)
        pd_empty = pd.DataFrame(index=index)
        url_series = pd_empty.merge(right=url_series,left_index=True , right_index=True, how='left')

    # TODO: 这里因为当第二个 pd.Series要插入到 df 中出现错误时，切记要从初始数据集的脏数据出发
    # 使用 url_series[col].index.duplicated()  或者  df.index.duplicated() 查看是否存在index重复的问题
    for col in url_series.columns:
        df[f"{ds_info.name}_{col}"] = url_series[col]
    return df

def create_dataset(dataset_name : str):
    ds_info = datasets_info[dataset_name]
    df_aim = load_finance_from_csv(ds_info)
    ds_metadata = {'prediction_length': ds_info.prediction_length,
                   'dim': ds_info.dim,
                   'freq':ds_info.freq,
    }
    func = slice_func[datasets_info[dataset_name].slice]
    if func != None: #表示存在窗口切割函数
        window_size = ds_info.prediction_length + ds_info.train_length
        target_slice , target_start = func(df_aim, window_size)
        ds_metadata['sample_size'] = len(target_slice)
        ds_metadata['num_step'] = window_size
        ds_metadata['start'] = target_start
        return target_slice,ds_metadata
    else: # 表示该数据集不进行切割
        # feat_static_cat = np.arange(ds_info.dim).astype(np.float32)
        ds_metadata['sample_size'] = 1
        ds_metadata['num_step'] = ds_info.num_time_steps
        ds_metadata['start'] = [pd.Timestamp(datasets_info[dataset_name].start_date
                                             ,freq = datasets_info[dataset_name].freq)
                                for _ in range(datasets_info[dataset_name].dim)]
        return np.transpose(df_aim.values) , ds_metadata



# 画原有序列
def plot_original(train_entry , test_entry , no):
    test_series = to_pandas(test_entry)
    train_series = to_pandas(train_entry)

    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 7))

    train_series.plot(ax=ax[0])
    ax[0].grid(which="both")
    ax[0].legend(["train series"], loc="upper left")

    test_series.plot(ax=ax[1])
    ax[1].axvline(train_series.index[-1], color='r')  # end of train dataset
    ax[1].grid(which="both")
    ax[1].legend(["test series", "end of train series"], loc="upper left")

    plt.savefig('pic/original_entry_{}.png'.format(no))

def createGluontsDataset(data_name):
    ds_info = datasets_info[data_name]
    # 获取所有目标序列，元信息
    target_slice, ds_metadata = create_dataset(finance_data_name)
    # print('feat_static_cat : ' , feat_static_cat , '  type:' ,type(feat_static_cat))
    train_ds = ListDataset([{FieldName.TARGET: target,
                             FieldName.START: start,}
                            for (target, start) in zip(target_slice[:, :-ds_metadata['prediction_length']],
                                                        ds_metadata['start'],
                                                        )],
                           freq=ds_metadata['freq'],
                           one_dim_target=False)

    test_ds = ListDataset([{FieldName.TARGET: target,
                            FieldName.START: start,
                            }
                           for (target, start) in zip(target_slice,
                                                       ds_metadata['start'],
                                                       )],
                          freq=ds_metadata['freq'],
                          one_dim_target=False)

    dataset = TrainDatasets(metadata=ds_metadata, train=train_ds, test=test_ds)
    with open('data_process/processed_data/{}_{}_{}_{}.pkl'.format(
            finance_data_name, '%s_DsSeries_%d'%(ds_info.slice,dataset.metadata['sample_size']),
            'train_%d'%ds_info.train_length, 'pred_%d'%ds_info.prediction_length,
    ), 'wb') as fp:
        pickle.dump(dataset, fp)

    print('当前数据集为: ', finance_data_name , '训练长度为 %d , 预测长度为 %d '%(ds_info.train_length , ds_info.prediction_length),
          '(切片之后)每个数据集样本数目为：'  , dataset.metadata['sample_size'])

if __name__ == '__main__':
    if 'lzl_shared_ssm' in os.getcwd():
        if 'data_process' in os.getcwd():
            os.chdir('..')
        print('------当前与处理数据的路径：', os.getcwd() ,'-------')
    else:
        print('处理数据的时候，请先进入finance的主目录')
        exit()

    finance_data_name = args.dataset
    createGluontsDataset(finance_data_name)




# pandas 设置 date_range的API
# time_index = pd.date_range(
#     start=ds_info.start_date,
#     freq=ds_info.freq,
#     periods=ds_info.num_time_steps,
# )
# df = df.set_index(time_index)


# 当 series_url 是一个 List的情况下
# def load_finance_from_csv(ds_info):
#     df = pd.DataFrame()
#     for url_no in range(len(ds_info.url)):
#         path, time_str  = ds_info.url[url_no], ds_info.timestamp[url_no]
#         if isinstance(ds_info.aim[url_no] , list): #当前数据集 有多个目标序列，或 传入的值为 List
#             aim = ds_info.aim[url_no]
#         elif isinstance(ds_info.aim[url_no] , str): #当前数据集 只有一个目标序列，传入的值是str
#             aim = [ds_info.aim[url_no]]
#         series_start = pd.Timestamp(ts_input=ds_info.start_date , freq=ds_info.freq)
#         series_end = series_start + (ds_info.num_time_steps-1)*series_start.freq
#         data_name = path.split('/')[-1].split('.')[0]
#         url_series = pd.read_csv(path ,sep=',' ,header=0 , parse_dates=[time_str])
#         if 'D' in ds_info.freq:
#             url_series[time_str] = pd.to_datetime(url_series[time_str].dt.date)
#         url_series.set_index(time_str ,inplace=True)
#         url_series = url_series.loc[series_start:series_end][aim]
#         if url_series.shape[0] < ds_info.num_time_steps:
#             #添加缺失的时刻(比如周末数据确实这样的)
#             index = pd.date_range(start=series_start, periods=ds_info.num_time_steps, freq=series_start.freq)
#             pd_empty = pd.DataFrame(index=index)
#             url_series = pd_empty.merge(right=url_series,left_index=True , right_index=True, how='left')
#
#         # 使用 url_series[col].index.duplicated()  或者  df.index.duplicated() 查看是否存在index重复的问题
#         for col in url_series.columns:
#             df[f"{data_name}_{col}"] = url_series[col]
#     return df