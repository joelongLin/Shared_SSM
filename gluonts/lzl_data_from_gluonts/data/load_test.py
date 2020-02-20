import mxnet as mx
import numpy as np
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
import pickle
import os
import tensorflow as tf
from gluonts import transform
from gluonts.transform import TransformedDataset
from gluonts.model.deepstate import DeepStateEstimator
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.lzl_deepstate.utils import reload_config
from gluonts.dataset.loader import  InferenceDataLoader

cl = tf.app.flags
# reload_model =  'logs/btc_eth/Dec_25_17:27:07_2019'
reload_model = ''
cl.DEFINE_string('reload_model' ,reload_model,'model to reload')
cl.DEFINE_string('logs_dir','logs/btc_eth','file to print log')

# network configuration
cl.DEFINE_integer('num_layers' ,2,'num of lstm cell layers')
cl.DEFINE_integer('num_cells' ,40 , 'hidden units size of lstm cell')
cl.DEFINE_string('cell_type' , 'lstm' , 'Type of recurrent cells to use (available: "lstm" or "gru"')
cl.DEFINE_float('dropout_rate' , 0.1 , 'Dropout regularization parameter (default: 0.1)')
cl.DEFINE_string('embedding_dimension' , '' , ' Dimension of the embeddings for categorical features')

# dataset configuration
cl.DEFINE_string('dataset' , 'electricity' , 'Name of the target dataset')
cl.DEFINE_string('freq','1H','Frequency of the data to train on and predict')
cl.DEFINE_integer('num_periods_to_train' ,4,'This is the length of the training time series')
cl.DEFINE_integer('prediction_length' , 168 , 'Length of the prediction horizon')
cl.DEFINE_bool('add_trend' , False , 'Flag to indicate whether to include trend component in the SSM')

# prediciton configuration
cl.DEFINE_integer('num_eval_samples', '100', 'Number of samples paths to draw when computing predictions')
cl.DEFINE_bool('scaling', True, 'whether to scale the target and observed')
cl.DEFINE_bool('use_feat_dynamic_real', False, 'Whether to use the ``feat_dynamic_real`` field from the data')
cl.DEFINE_bool('use_feat_static_cat', True, 'Whether to use the ``feat_static_cat`` field from the data')
cl.DEFINE_string('cardinality' , '2' , 'Number of values of each categorical feature.')

#train configuration
cl.DEFINE_integer('epochs' , 25 , 'Number of epochs that the network will train (default: 1).')
cl.DEFINE_bool('shuffle' , False ,'whether to shuffle the train dataset')
cl.DEFINE_integer('batch_size' ,  32 , 'Numbere of examples in each batch')
cl.DEFINE_integer('num_batches_per_epoch' , 200 , 'Numbers of batches at each epoch')
cl.DEFINE_float('learning_rate' , 0.001 , 'Initial learning rate')

config = cl.FLAGS
config = reload_config(config)

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
        num_periods_to_train = config.num_periods_to_train, # default : 4
        trainer = Trainer(epochs=config.epochs,batch_size=1,num_batches_per_epoch=100000, hybridize=False),
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
            dtype=estimator.float_type,
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