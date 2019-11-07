import mxnet as mx
import numpy as np
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.dataset.util import to_pandas
from gluonts.dataset.loader import TrainDataLoader
from gluonts.gluonts_tqdm import tqdm
from gluonts.support.util import get_hybrid_forward_input_names
import pickle
import os



from gluonts.model.deepstate import DeepStateEstimator
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions

if ('/lzl_test' not in os.getcwd()):
     os.chdir('gluonts/lzl_test')
     print('change os dir : ',os.getcwd())
elec_ds = get_dataset("electricity", regenerate=False)

estimator = DeepStateEstimator(
        freq = elec_ds.metadata.freq,
        prediction_length = elec_ds.metadata.prediction_length*7,
        # add_trend = False,
        # past_length =2 * elec_ds.metadata.prediction_length,
        num_periods_to_train = 2,
        trainer = Trainer(epochs=25,hybridize=False),
        num_layers = 2,
        num_cells = 40,
        cell_type = "lstm",
        num_eval_samples = 100,
        dropout_rate = 0.1,
        use_feat_dynamic_real = False,
        use_feat_static_cat = True,
        # cardinality=[12,13,14,15],
        cardinality=[321],
        scaling = True,
)



predictor = estimator.train(elec_ds.train)

# predictor_path = 'predictor_{}.pkl'.format(estimator.embedding_dimension)
# with open(predictor_path , "wb") as fp:
#         pickle.dump(predictor, fp)

# with open('predictor_20.pkl' , "rb") as fp:
#         predictor = pickle.load(fp)


# forecast_it, ts_it = make_evaluation_predictions(
#     dataset=elec_ds.test,  # test dataset
#     predictor=predictor,  # predictor
#     num_eval_samples=100,  # number of sample paths we want for evaluation
# )


# forecasts = list(forecast_it) #(sample_size , mc_sample_size , prediction_len)
# exit()

# forecasts_path = 'forecast_electricity_{}.pkl'.format( 20)
# with open(forecasts_path , "wb") as fp:
#         pickle.dump(forecasts, fp)
# tss = list(ts_it)

# first entry of the time series list
# ts_entry = tss[0]
# first 5 values of the time series (convert from pandas to numpy)
# np.array(ts_entry[:5]).reshape(-1,)

# test_ds_entry = next(iter(elec_ds.test))
