import numpy as np
import pandas as pd
from typing import List, NamedTuple, Optional
from gluonts.dataset.common import ListDataset
from gluonts.dataset.common import TrainDatasets
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import to_pandas
from gluonts import transform
import matplotlib.pyplot as plt
from gluonts.transform import *
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.lzl_deepstate.model.issm import CompositeISSM
from gluonts.dataset.loader import  InferenceDataLoader
from tqdm import tqdm
import pickle

root='/home/lzl/pycharm/gluon/gluonts/lzl_finance/data_process/raw_data/'
class Dataset(NamedTuple):
    name: str
    url: list
    num_series: int
    num_time_steps: int
    train_length:int
    prediction_length: int
    slice:bool
    freq: str
    start_date: str
    agg_freq: Optional[str] = None


datasets_info = {
    "btc_eth": Dataset(
        name="btc_eth",
        url=[root+'eth.csv', root+'btc.csv'],
        num_series=2,
        num_time_steps=503,
        train_length=60,
        prediction_length=1,
        slice=True,
        start_date="2018-08-02",
        freq="1D",
        agg_freq=None,
    ),
    "btc_eth_1h" : Dataset(
        name="btc_eth_1h",
        url=[root+'eth1h.csv', root+'btc1h.csv'],
        num_series=2,
        num_time_steps=12232,
        train_length=24*7,
        prediction_length=1,
        slice=True,
        start_date="2018-08-02",
        freq="1H",
        agg_freq=None,
    ),
}

def load_dataframe_from_csv(ds_info):
    time_index = pd.date_range(
        start=ds_info.start_date,
        freq=ds_info.freq,
        periods=ds_info.num_time_steps,
    )
    df = pd.DataFrame()
    for path in ds_info.url:
        url_name = path.split('/')[-1].split('.')[0]
        url_series = pd.read_csv(path ,sep=',' ,header=0).iloc[:ds_info.num_time_steps,1]
        df[url_name] = url_series
    df = df.set_index(time_index)
    return df

def slice_df_with_window(
    dataframe:pd.DataFrame
    ,window_size:int ):
    data = dataframe.values
    timestamp = dataframe.index
    target_slice,target_start, feat_static_cat ,label = [], [] ,[] ,[]
    _ , num_series = data.shape
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

def create_dataset(dataset_name : str):
    ds_info = datasets_info[dataset_name]
    df_aim = load_dataframe_from_csv(ds_info)
    ds_medadata = {'prediction_length': ds_info.prediction_length,
                   'freq':ds_info.freq,
    }
    if datasets_info[dataset_name].slice:
        window_size = ds_info.prediction_length + ds_info.train_length
        target_slice , target_start,feat_static_cat \
            = slice_df_with_window(df_aim ,window_size)
        ds_medadata['num_series'] = len(target_slice)
        ds_medadata['num_step'] = window_size
        ds_medadata['start'] = target_start
        return target_slice,ds_medadata,feat_static_cat
    else:
        feat_static_cat = np.arange(ds_info.num_series).astype(np.float32)
        ds_medadata['num_series'] = ds_info.num_series
        ds_medadata['num_step'] = ds_info.num_time_steps
        ds_medadata['start'] = [pd.Timestamp(datasets_info[dataset_name].start_date
                                             ,freq = datasets_info[dataset_name].freq)
                                for _ in range(datasets_info[dataset_name].num_series)]
        return np.transpose(df_aim.values) , ds_medadata ,feat_static_cat


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
            #  InstanceSplitter(
            #     target_field=FieldName.TARGET,
            #     is_pad_field=FieldName.IS_PAD,
            #     start_field=FieldName.START,
            #     forecast_start_field=FieldName.FORECAST_START,
            #     train_sampler=ExpectedNumInstanceSampler(num_instances=1),
            #     past_length=ds_info.train_length,
            #     future_length= ds_info.prediction_length,
            #     output_NTC = False,
            #     time_series_fields = [
            #         FieldName.FEAT_AGE,
            #         'seasonal_indicators',
            #         FieldName.OBSERVED_VALUES
            #     ],
            #  )
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
    finance_data_name = 'btc_eth_1h'
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


    with open('../../lzl_deepstate/data/groundtruth_{}_{}_{}.pkl'.format(
            finance_data_name, ds_info.train_length, ds_info.prediction_length,
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
    with open('../../lzl_deepstate/data/train_{}_{}_{}.pkl'.format(
            finance_data_name,ds_info.train_length,ds_info.prediction_length,
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
    with open('../../lzl_deepstate/data/test_{}_{}_{}.pkl'.format(
            finance_data_name, ds_info.train_length, ds_info.prediction_length,
    ), 'wb') as fp:
        pickle.dump(test_all_result, fp)
    print('已获取全部的测试数据')