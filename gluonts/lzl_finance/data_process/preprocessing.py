from enum import Enum, unique
import numpy as np
import pandas as pd
from typing import List, NamedTuple, Optional ,Union
from gluonts.dataset.common import ListDataset
from gluonts.dataset.common import TrainDatasets
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import to_pandas
from gluonts import transform
import matplotlib.pyplot as plt
from gluonts.transform import (Chain,
                               AsNumpyArray,
                               ExpandDimArray,
                               AddObservedValuesIndicator,
                               AddTimeFeatures,
                               AddAgeFeature,
                               VstackFeatures,
                               CanonicalInstanceSplitter,
                               TestSplitSampler)
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.lzl_deepstate.model.issm import CompositeISSM
from gluonts.dataset.loader import  InferenceDataLoader
from tqdm import tqdm
import pickle
import os
if 'lzl_finance'  in os.getcwd():
    if 'data_process' in os.getcwd():
        os.chdir('..')
        print('当前的路径如下：' , os.getcwd())
else:
    print('处理数据的时候，请先进入finance的主目录')
    exit(-1)
root='data_process/raw_data/'
class Dataset(NamedTuple):
    name: str  # 该数据集的名称
    url: List  #存放该数据集的路径的list
    timestampe : List[str]  #表明这些 url 里面对应的 表示时刻的名称 如: eth.csv 中的 beijing_time  或者 btc.csv 中的Date
    num_series: int  #该数据集存在多少条序列
    num_time_steps: int # 该数据集对应的序列的数量
    train_length:int    #训练长度
    prediction_length: int  #预测长度
    slice:str  #是否进行切片('no' ,'lap' ,'nolap')，切片的长度是train_length + prediction_length
    freq: str   # 该序列时间步的大小
    start_date: str #序列开始的时间
    aim : List[Union[List[str] , str]] # 序列的目标特征，包含的数量特征应该与 url， timestamp一致
    feat_dynamic_cat: Optional[str] = None  #表明该序列类别的 (seq_length , category)


datasets_info = {
    "btc_eth": Dataset(
        name="btc_eth",
        url=[root+'eth.csv', root+'btc.csv'],
        timestampe = ['beijing_time' , 'Date'] ,
        num_series=2,
        num_time_steps=503,
        train_length=60,
        prediction_length=1,
        slice='lap',
        start_date="2018-08-02",
        aim=[['close'] , 'close'], # 这样子写只是为了测试预处理程序是否强大
        freq="1D",
    ),
    "btc_eth_1h" : Dataset(
        name="btc_eth_1h",
        url=[root+'eth1h.csv', root+'btc1h.csv'],
        timestampe = ['beijing_time' , 'beijing_time'],
        num_series=2,
        num_time_steps=12232,
        train_length=24*7,
        prediction_length=1,
        slice='lap',
        start_date="2018-08-02",
        aim=['close'],
        freq="1H",
    ),
}
# 切割完之后， 除了目标序列target_slice 之外
# 需要给定开始时间target_start， 属于哪个类feat_static_cat
def slice_df_overlap(
    dataframe:pd.DataFrame
    ,window_size:int
):
    data = dataframe.values
    timestamp = dataframe.index
    target_slice,target_start, feat_static_cat ,label = [], [] ,[] ,[]
    _ , num_series = data.shape
    # feat_static_cat 特征值，还是按照原序列进行标注
    for j in range(num_series):
        for i in range(window_size - 1, len(data)):
            feat_static_cat.append(j)
            label.append(data[i, j])
            a = data[(i - window_size + 1):(i + 1), j]
            target_slice.append(a)
            target_start.append(timestamp[i-window_size+1])

    target_slice,feat_static_cat, label = \
        np.array(target_slice), np.array(feat_static_cat ,dtype=np.float32) , np.array(label)
    return target_slice, target_start ,feat_static_cat

def slice_df_nolap(
    dataframe:pd.DataFrame
    ,window_size:int
):
    data = dataframe.values
    timestamp = dataframe.index
    target_slice, target_start, feat_static_cat, label = [], [], [], []
    _, num_series = data.shape
    for j in range(num_series):
        # 这里面 根据窗口长度选择长度
        # TODO: 暂时没空写这个方法，后面有空写上
        for i in range((len(data) // window_size)+1):
            pass
    pass

slice_func = {
    'lap' : slice_df_overlap,
    'nolap' : slice_df_nolap
}
def load_finance_from_csv(ds_info):
    df = pd.DataFrame()
    for url_no in range(len(ds_info.url)):
        path, time_str  = ds_info.url[url_no], ds_info.timestampe[url_no]
        if isinstance(ds_info.aim[url_no] , list): #当前数据集 有多个目标序列，或 传入的值为 List
            aim = ds_info.aim[url_no]
        elif isinstance(ds_info.aim[url_no] , str): #当前数据集 只有一个目标序列，传入的值是str
            aim = [ds_info.aim[url_no]]
        series_start = pd.Timestamp(ts_input=ds_info.start_date , freq=ds_info.freq)
        series_end = series_start + (ds_info.num_time_steps-1)*series_start.freq
        data_name = path.split('/')[-1].split('.')[0]
        url_series = pd.read_csv(path ,sep=',' ,header=0 , parse_dates=[time_str])
        # TODO: 这里要注意不够 general 只在适合 freq='D' 的情况
        if 'D' in ds_info.freq:
            url_series[time_str] = url_series[time_str].dt.date
        url_series.set_index(time_str ,inplace=True)
        url_series = url_series.loc[series_start:series_end][aim]
        # TODO: 这里因为当第二个 pd.Series要插入到 df 中出现错误时，切记要从初始数据集的脏数据出发
        # 使用 url_series[col].index.duplicated()  或者  df.index.duplicated() 查看是否存在index重复的问题
        for col in url_series:
            df[f"{data_name}_{col}"] = url_series[col]
    return df


def create_dataset(dataset_name : str):
    ds_info = datasets_info[dataset_name]
    df_aim = load_finance_from_csv(ds_info)
    ds_metadata = {'prediction_length': ds_info.prediction_length,
                   'freq':ds_info.freq,
    }
    func = slice_func[datasets_info[dataset_name].slice]
    if func != None: #表示存在窗口切割函数
        window_size = ds_info.prediction_length + ds_info.train_length
        target_slice , target_start , feat_static_cat \
            = func(df_aim, window_size)
        ds_metadata['num_series'] = len(target_slice)
        ds_metadata['num_step'] = window_size
        ds_metadata['start'] = target_start
        return target_slice,ds_metadata,feat_static_cat
    else: # 表示该数据集不进行切割
        feat_static_cat = np.arange(ds_info.num_series).astype(np.float32)
        ds_metadata['num_series'] = ds_info.num_series
        ds_metadata['num_step'] = ds_info.num_time_steps
        ds_metadata['start'] = [pd.Timestamp(datasets_info[dataset_name].start_date
                                             ,freq = datasets_info[dataset_name].freq)
                                for _ in range(datasets_info[dataset_name].num_series)]
        return np.transpose(df_aim.values) , ds_metadata ,feat_static_cat

def create_transformation(dataset_name : str):
    ds_info = datasets_info[dataset_name]
    return Chain(
         [
            AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
            AsNumpyArray(field=FieldName.TARGET, expected_ndim=1),
            # gives target the (1, T) layout
            ExpandDimArray(field=FieldName.TARGET, axis=0),
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            # Unnormalized seasonal features
            AddTimeFeatures(
                time_features=CompositeISSM.seasonal_features(ds_info.freq),
                pred_length=ds_info.prediction_length,
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field="seasonal_indicators",
            ),
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features_from_frequency_str(ds_info.freq),
                pred_length=ds_info.prediction_length,
            ),
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=ds_info.prediction_length,
                log_scale=True,
            ),
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
            ),

            CanonicalInstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                instance_sampler=TestSplitSampler(),
                time_series_fields=[
                    FieldName.FEAT_TIME,
                    "seasonal_indicators",
                    FieldName.OBSERVED_VALUES,
                ],
                allow_target_padding=True,
                instance_length=ds_info.train_length,
                use_prediction_features=True,
                prediction_length=ds_info.prediction_length,
            ),
        ]
    )

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

if __name__ == '__main__':
    finance_data_name = 'btc_eth'
    ds_info = datasets_info[finance_data_name]

    # 获取所有目标序列，元信息
    target_slice , ds_metadata ,feat_static_cat= create_dataset(finance_data_name)
    transformation = create_transformation(finance_data_name)
    # print('feat_static_cat : ' , feat_static_cat , '  type:' ,type(feat_static_cat))
    train_ds = ListDataset([{FieldName.TARGET: target,
                             FieldName.START: start,
                             FieldName.FEAT_STATIC_CAT: fsc}
                            for (target, start, fsc) in zip(target_slice[:, :-ds_metadata['prediction_length']],
                                                                 ds_metadata['start'],
                                                                 feat_static_cat)],
                           freq=ds_metadata['freq'])

    test_ds = ListDataset([{FieldName.TARGET: target,
                            FieldName.START: start,
                            FieldName.FEAT_STATIC_CAT: fsc}
                           for (target, start, fsc) in zip(target_slice,
                                                                ds_metadata['start'],
                                                                feat_static_cat)],
                          freq=ds_metadata['freq'])

    groundTruth = TrainDatasets(metadata=ds_metadata, train=train_ds, test=test_ds)


    with open('processed_data/groundtruth_{}_{}_{}_{}.pkl'.format(
            finance_data_name, ds_info.slice,
            ds_info.train_length, ds_info.prediction_length,
    ), 'wb') as fp:
        pickle.dump(groundTruth, fp)

    it_test_ds = iter(test_ds)

    train_tf = transformation(iter(train_ds), is_train=True)
    # 在这里试着修改一下，依然选择用it_train_ds, 结果发现是对的
    test_tf = transformation(iter(train_ds) , is_train=False)
    train_inputs_name = ['feat_static_cat', 'past_observed_values', 'past_seasonal_indicators', 'past_time_feat', 'past_target']
    test_inputs_name = ['feat_static_cat', 'past_observed_values'
                     , 'past_seasonal_indicators', 'past_time_feat'
                     , 'past_target', 'future_seasonal_indicators'
                     , 'future_time_feat' , 'forecast_start']

    train_all_result = []
    with tqdm(train_tf) as it:
        for batch_no, data_entry in enumerate(it, start=1):
            # inputs = [data_entry[k] for k in input_names]

            inputs = [data_entry[k] for k in train_inputs_name]
            train_all_result.append(inputs)
            # print('当前第 ', batch_no, '正输入 all_data 中')

    print('train_all_result 的长度大小为：', len(train_all_result))
    with open('processed_data/train_{}_{}_{}_{}.pkl'.format(
            finance_data_name , ds_info.slice ,
            ds_info.train_length , ds_info.prediction_length,
    ), 'wb') as fp:
        pickle.dump(train_all_result, fp)
    print('已获取全部的训练数据')

    test_all_result = []
    with tqdm(test_tf) as it:
        for batch_no, data_entry in enumerate(it, start=1):
            inputs = [data_entry[k] for k in test_inputs_name]
            test_all_result.append(inputs)

    print('test_all_result 的长度大小为：', len(test_all_result))
    ds_info = datasets_info[finance_data_name]
    with open('processed_data/test_{}_{}_{}_{}.pkl'.format(
            finance_data_name, ds_info.slice,
            ds_info.train_length, ds_info.prediction_length,
    ), 'wb') as fp:
        pickle.dump(test_all_result, fp)
    print('已获取全部的测试数据')

# pandas 设置 date_range的API
# time_index = pd.date_range(
#     start=ds_info.start_date,
#     freq=ds_info.freq,
#     periods=ds_info.num_time_steps,
# )
# df = df.set_index(time_index)