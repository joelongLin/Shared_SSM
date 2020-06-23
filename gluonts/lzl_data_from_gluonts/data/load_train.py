import pickle
import os
import numpy as np
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.dataset.util import to_pandas
from gluonts.dataset.loader import TrainDataLoader
from gluonts.gluonts_tqdm import tqdm
from gluonts.support.util import get_hybrid_forward_input_names
from data_utils import whetherInputInList
from pathlib import Path



from gluonts.model.deepstate import DeepStateEstimator
from gluonts.trainer import Trainer


if ('/lzl_data_from_gluonts' not in os.getcwd()):
     os.chdir('gluonts/lzl_data_from_gluonts')
     print('change os dir : ',os.getcwd())
dataset_name = "electricity"
elec_ds = get_dataset(dataset_name, regenerate=False)

#注意 这里的num_periods_to_train  表示 训练窗口与预测窗口的倍数关系
estimator = DeepStateEstimator(
        freq = elec_ds.metadata.freq,
        prediction_length = elec_ds.metadata.prediction_length*7,
        cardinality = [321],
        add_trend = False,
        num_periods_to_train = 3, # default : 4
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

train_iter = TrainDataLoader(
            dataset=elec_ds.train,
            transform = estimator.create_transformation(),
            batch_size= estimator.trainer.batch_size,
            num_batches_per_epoch = estimator.trainer.num_batches_per_epoch,
            ctx=estimator.trainer.ctx,
            shuffle_for_training= False, # 源码中设置的是 打乱training 的顺序，在这里设置成 False ，也就是不会打乱顺序
            dtype=estimator.float_type,
        )
trained_net = estimator.create_training_network()
input_names=get_hybrid_forward_input_names(trained_net)

all_result =[]
with tqdm(train_iter) as it:
    for batch_no, data_entry in enumerate(it, start=1):
        # inputs = [data_entry[k] for k in input_names]

        inputs = [np.squeeze(data_entry[k].asnumpy(), axis=0) for k in input_names]
        if not whetherInputInList(inputs, all_result, -1):
            all_result.append(inputs)
            if batch_no % 1 == 0:
                print('当前第 ', batch_no, '正输入 all_data 中')
        else:
            print('输入结束了')
            break
        continue


with open('../../lzl_deepstate/data/train_electricity_{}_{}.pkl'.format(estimator.past_length,estimator.prediction_length)
        , 'wb') as fp:
    pickle.dump(all_result, fp)
print('已获取全部的数据')

