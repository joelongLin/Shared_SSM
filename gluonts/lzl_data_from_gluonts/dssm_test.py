import mxnet as mx
import numpy as np
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.dataset.util import to_pandas
from gluonts.dataset.loader import TrainDataLoader
from gluonts.gluonts_tqdm import tqdm
from gluonts.support.util import get_hybrid_forward_input_names
import pickle

elec_ds = get_dataset("electricity", regenerate=False)

from gluonts.model.deepstate import DeepStateEstimator
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions

estimator = DeepStateEstimator(
        freq = elec_ds.metadata.freq,
        prediction_length = elec_ds.metadata.prediction_length,
        add_trend = False,
        past_length =2 * elec_ds.metadata.prediction_length,
        trainer = Trainer(epochs=25,batch_size=128,num_batches_per_epoch=10, hybridize=False),
        num_layers = 2,
        num_cells = 40,
        cell_type = "lstm",
        num_eval_samples = 100,
        dropout_rate = 0.1,
        use_feat_dynamic_real = False,
        use_feat_static_cat = False,
        embedding_dimension = 20,
        scaling = True,
)
trans = estimator.create_transformation()
net = estimator.create_training_network()

training_data_loader = TrainDataLoader(
            dataset=elec_ds.train,
            transform=trans,
            batch_size=estimator.trainer.batch_size,
            num_batches_per_epoch=estimator.trainer.num_batches_per_epoch,
            ctx=estimator.trainer.ctx,
            float_type=estimator.float_type,
        )

input_names = get_hybrid_forward_input_names(net)

print(input_names)
exit()
all_train_data = []
with tqdm(training_data_loader) as it:
        for batch_no, data_entry in enumerate(it, start=1):

                inputs = {k :data_entry[k] for k in input_names}
                all_train_data.append(inputs)

all_train_data_path = './all_train_data.pkl'
with open(all_train_data_path, 'wb') as fp:
    pickle.dump(all_train_data, fp)


# predictor = estimator.train(elec_ds.train)


# forecast_it, ts_it = make_evaluation_predictions(
#     dataset=elec_ds.test,  # test dataset
#     predictor=predictor,  # predictor
#     num_eval_samples=100,  # number of sample paths we want for evaluation
# )
#
# forecasts = list(forecast_it)
# tss = list(ts_it)
