import mxnet as mx
import numpy as np
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.dataset.util import to_pandas
from gluonts.dataset.loader import TrainDataLoader
from gluonts.gluonts_tqdm import tqdm
from gluonts.support.util import get_hybrid_forward_input_names
from pathlib import Path
import pickle
import os



from gluonts.model.deepstate import DeepStateEstimator
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions

if ('/lzl_test' not in os.getcwd()):
     os.chdir('gluonts/lzl_test')
     print('change os dir : ',os.getcwd())
dataset_name = "electricity"
elec_ds = get_dataset(dataset_name, regenerate=False)

# estimator = DeepStateEstimator(
#         freq = elec_ds.metadata.freq,
#         prediction_length = elec_ds.metadata.prediction_length*7,
#         cardinality = [321],
#         add_trend = False,
#         num_periods_to_train = 4, # default : 4
#         trainer = Trainer(epochs=25,batch_size=1,num_batches_per_epoch=100000, hybridize=False),
#         num_layers = 2,
#         num_cells = 40,
#         cell_type = "lstm",
#         num_eval_samples = 100,
#         dropout_rate = 0.1,
#         use_feat_dynamic_real = False,
#         use_feat_static_cat = True,
#         scaling = True,
# )

# predictor = estimator.train(elec_ds.train)
#
# predictor_path = 'predictor/{}_{}_{}'.format(
#         dataset_name,estimator.past_length ,estimator.prediction_length)
# if not os.path.isdir(predictor_path):
#     os.mkdir(predictor_path)
#

# predictor.serialize(Path(predictor_path))



# TODO 开始执行第二段
from gluonts.model.predictor import Predictor
predictor_deserialized = Predictor.deserialize(Path("predictor/electricity_336_168"))
print(predictor_deserialized.input_names)
# predictor_deserialized.batch_size = 1

forecast_it, ts_it = make_evaluation_predictions(
    dataset=elec_ds.test,  # test dataset
    predictor=predictor_deserialized,  # predictor
    num_eval_samples=100,  # number of sample paths we want for evaluation
)


forecasts = list(forecast_it) #(sample_size , mc_sample_size , prediction_len)
tss = list(ts_it)

forecasts_path = 'tmp/predict_result/forecasts_electricity_{}_{}.pkl'.format(
                predictor_deserialized.prediction_net.past_length
                ,predictor_deserialized.prediction_length
            )
tss_path = 'tmp/predict_result/tss_electricity_{}_{}.pkl'.format(
                predictor_deserialized.prediction_net.past_length
                ,predictor_deserialized.prediction_length
            )
with open(forecasts_path , "wb") as fp:
        pickle.dump(forecasts, fp)

with open(tss_path, "wb") as fp:
    pickle.dump(tss, fp)