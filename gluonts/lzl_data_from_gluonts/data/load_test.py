import mxnet as mx
import numpy as np
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
import pickle
import os

from gluonts import transform
from gluonts.transform import TransformedDataset
from gluonts.model.deepstate import DeepStateEstimator
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.lzl_deepstate.utils import get_image_config,reload_config
from gluonts.dataset.loader import  InferenceDataLoader

config = get_image_config()
config = reload_config(config.FLAGS)

if ('/lzl_data_from_gluonts' not in os.getcwd()):
     os.chdir('gluonts/lzl_data_from_gluonts')
     print('change os dir : ',os.getcwd())
dataset_name = "electricity"
elec_ds = get_dataset(dataset_name, regenerate=False)

estimator = DeepStateEstimator(
        freq = elec_ds.metadata.freq,
        prediction_length = elec_ds.metadata.prediction_length*7,
        cardinality = [321],
        add_trend = False,
        num_periods_to_train = 4, # default : 4
        trainer = Trainer(epochs=25,batch_size=1,num_batches_per_epoch=100000, hybridize=False),
        num_layers = 3,
        num_cells = 40,
        cell_type = "lstm",
        num_eval_samples = 100,
        dropout_rate = 0.1,
        use_feat_dynamic_real = False,
        use_feat_static_cat = True,
        scaling = True,
)

def truncate_target(data):
    data = data.copy()
    target = data["target"]
    assert (
            target.shape[-1] >= config.prediction_length
    )  # handles multivariate case (target_dim, history_length)
    data["target"] = target[..., :-config.prediction_length]  # 对target序列进行裁剪为排除预测域
    return data

dataset_trunc = TransformedDataset(
        elec_ds.test, transformations=[transform.AdhocTransform(truncate_target)]
)
inference_data_loader = InferenceDataLoader(
            dataset_trunc,
            estimator.create_transformation(),
            estimator.trainer.batch_size,
            ctx=estimator.trainer.ctx,
            float_type=estimator.float_type,
        )
input_names =['feat_static_cat', 'past_observed_values'
    , 'past_seasonal_indicators', 'past_time_feat'
    , 'past_target', 'future_seasonal_indicators', 'future_time_feat']
no_batch = 0
test_all_result = []; input_names.append('forecast_start')
for batch in inference_data_loader:  # 这里会产生 batch_size 个 sample 组成的batch
    no_batch += 1;

    inputs = [np.squeeze(batch[k].asnumpy(), axis=0)
              if isinstance(batch[k] , mx.nd.NDArray)
              else batch[k][0]
              for k in input_names]
    test_all_result.append(inputs)
    print('当前第 ' ,no_batch , '正输入 test_all_data(%d 存量) 中'%(len(test_all_result)) )
    continue


print('test_all_result 的长度大小为：', len(test_all_result))
with open('../../lzl_deepstate/data/test_electricity_{}_{}.pkl'.format(
    estimator.past_length, estimator.prediction_length
), 'wb') as fp:
    pickle.dump(test_all_result, fp)
print('已获取全部的测试数据')
exit()