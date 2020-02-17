from .preprocessing import datasets_info
import numpy as np
from gluonts.dataset.field_names import FieldName
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

def create_transformation(dataset_name : str):
    info = datasets_info[dataset_name]
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
                time_features=CompositeISSM.seasonal_features(info.freq),
                pred_length=info.prediction_length,
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field="seasonal_indicators",
            ),
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features_from_frequency_str(info.freq),
                pred_length=info.prediction_length,
            ),
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=info.prediction_length,
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
                instance_length=info.train_length,
                use_prediction_features=True,
                prediction_length=info.prediction_length,
            ),
        ]
    )

# 此函数将只用于调试
if __name__ == '__main__':
    # it_test_ds = iter(test_ds)
    #
    # train_tf = transformation(iter(train_ds), is_train=True)
    # # 在这里试着修改一下，依然选择用it_train_ds, 结果发现是对的
    # test_tf = transformation(iter(train_ds), is_train=False)
    # train_inputs_name = ['feat_static_cat', 'past_observed_values', 'past_seasonal_indicators', 'past_time_feat',
    #                      'past_target']
    # test_inputs_name = ['feat_static_cat', 'past_observed_values'
    #     , 'past_seasonal_indicators', 'past_time_feat'
    #     , 'past_target', 'future_seasonal_indicators'
    #     , 'future_time_feat', 'forecast_start']
    #
    # train_all_result = []
    # with tqdm(train_tf) as it:
    #     for batch_no, data_entry in enumerate(it, start=1):
    #         # inputs = [data_entry[k] for k in input_names]
    #
    #         inputs = [data_entry[k] for k in train_inputs_name]
    #         train_all_result.append(inputs)
    #         # print('当前第 ', batch_no, '正输入 all_data 中')
    #
    # print('train_all_result 的长度大小为：', len(train_all_result))
    # with open('processed_data/train_{}_{}_{}_{}.pkl'.format(
    #         finance_data_name, ds_info.slice,
    #         ds_info.train_length, ds_info.prediction_length,
    # ), 'wb') as fp:
    #     pickle.dump(train_all_result, fp)
    # print('已获取全部的训练数据')
    #
    # test_all_result = []
    # with tqdm(test_tf) as it:
    #     for batch_no, data_entry in enumerate(it, start=1):
    #         inputs = [data_entry[k] for k in test_inputs_name]
    #         test_all_result.append(inputs)
    #
    # print('test_all_result 的长度大小为：', len(test_all_result))
    # ds_info = datasets_info[finance_data_name]
    # with open('processed_data/test_{}_{}_{}_{}.pkl'.format(
    #         finance_data_name, ds_info.slice,
    #         ds_info.train_length, ds_info.prediction_length,
    # ), 'wb') as fp:
    #     pickle.dump(test_all_result, fp)
    # print('已获取全部的测试数据')
    pass