# -*- coding: UTF-8 -*-
# author : joelonglin
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

# Create transformation for SSSM input
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