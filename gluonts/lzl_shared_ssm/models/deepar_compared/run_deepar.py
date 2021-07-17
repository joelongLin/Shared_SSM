import pandas as pd
import os
import sys
# 使用bash运行，目录再 pycharm/gluon
if '/lzl_shared_ssm' not in os.getcwd():
    os.chdir('gluonts/lzl_shared_ssm')
    sys.path.insert(0, '../..')


from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.lzl_shared_ssm.utils import create_dataset_if_not_exist ,time_format_from_frequency_str, add_time_mark_to_file
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="deepar")
parser.add_argument('-t' ,'--target' , type=str , help='target series', default='btc,eth')
parser.add_argument('-st','--start', type=str, help='start time of the dataset',default='2018-08-02')
parser.add_argument('-f'  ,'--freq' , type = str , help = 'frequency of dataset' , default='1D')
parser.add_argument('-T','--timestep', type=int, help='length of the dataset', default=503)
parser.add_argument('-pr'  ,'--pred' , type = int , help = 'length of prediction range' , default=1)
parser.add_argument('-pa'  ,'--past' , type = int , help = 'length of training range' , default=30)
parser.add_argument('-s'  ,'--slice' , type = str , help = 'type of sliding dataset' , default='overlap')
parser.add_argument('-bs'  ,'--batch_size' , type = int , help = 'batch size' , default=32)
parser.add_argument('-bn'  ,'--batch_num' , type = int , help = 'batch number per epoch' , default=15)
parser.add_argument('-ep'  ,'--epochs' , type = int , help = 'epochs to train' , default=100)
args = parser.parse_args()
# change "_"  in start
if("_" in args.start):
    args.start = args.start.replace("_", " ")

item = args.target
start = args.start
freq = args.freq
timestep = args.timestep
pred = args.pred
past = args.past
slice_style = args.slice
batch_size = args.batch_size
batch_num = args.batch_num
epochs = args.epochs
ds_name_prefix = 'data_process/processed_data/{}_{}_{}_{}.pkl'

result_params = '_'.join([
                                'freq(%s)'%(freq),
                               'past(%d)'%(past)
                                ,'pred(%d)'%(pred)
                             ]
                        )


if __name__ == '__main__':
    # 导入 target 
    if slice_style == 'overlap':
        series = timestep - past - pred + 1
        print('每个数据集的序列数量为 ', series)
    elif slice_style == 'nolap':
        series = timestep // (past + pred)
        print('每个数据集的序列数量为 ', series)
    else:
        series = 1
        print('每个数据集的序列数量为 ', series ,'情景为长单序列')
    result_root_path  = 'evaluate/results/{}_length({})_slice({})_past({})_pred({})'.format(item.replace(',' ,'_'), timestep , slice_style ,past , pred)
    if not os.path.exists(result_root_path):
        os.makedirs(result_root_path)
    forecast_result_saved_path = os.path.join(result_root_path,'deepar_'  + result_params + '.pkl')
    forecast_result_saved_path = add_time_mark_to_file(forecast_result_saved_path)

    # 目标序列的数据路径
    target_path = {ds_name: ds_name_prefix.format(
        '%s_start(%s)_freq(%s)' % (ds_name, start,freq), '%s_DsSeries_%d' % (slice_style, series),
        'train_%d' % past, 'pred_%d' % pred
    ) for ds_name in item.split(',')}

    create_dataset_if_not_exist(
        paths=target_path, start=start, past_length=past
        , pred_length=pred, slice=slice_style
        , timestep=timestep, freq=freq
    )

    if not os.path.exists(forecast_result_saved_path):
        
        # 由于 target 应该都是 dim = 1 只是确定有多少个 SSM 而已
        target_ds = None;
        for item_name in target_path:
            item = target_path[item_name]
            print('导入数据 : {}'.format(item))
            with open(item, 'rb') as fp:
                item_ds = pickle.load(fp)
                assert item_ds.metadata['dim'] == 1, 'target 序列的维度都应该为1'
                # 合并数据集
                if target_ds == None:
                    target_ds = item_ds.test
                else:
                    target_ds.list_data += item_ds.test.list_data

        
       
                
        deepar_estimator = DeepAREstimator(
            freq=freq, 
            prediction_length=pred,
            context_length=past,
            lags_seq=[1],
            trainer = Trainer(
                epochs=epochs,
                batch_size=batch_size*len(target_path),
                num_batches_per_epoch=batch_num,
                learning_rate=0.001
            )
        )
        #由于 target.tet里面的输入都多了一维，在这里我进行squeeze操作
        for i in range(len(target_ds.list_data)):
            process_target = target_ds.list_data[i]["target"].squeeze()                
            target_ds.list_data[i]["target"] = process_target;
        
        from gluonts.dataset.common import ProcessDataEntry
        target_ds.process = ProcessDataEntry(freq , True)
            

        deepar_predictor = deepar_estimator.train(target_ds)
        # from pathlib import Path
        # predictor.serialize(Path("logs/deepar\(btc_eth\)"))

        
        from gluonts.evaluation.backtest import make_evaluation_predictions
        
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=target_ds,  # test dataset
            predictor=deepar_predictor,  # predictor
            num_samples=100,  # number of sample paths we want for evaluation
        )
            # forecasts = list(forecast_it)

            
            # 将forecasts的内容都一个一个遍历出来
            
        sample_forecasts = np.concatenate(
            [np.expand_dims(forecast.samples , axis=0) for forecast in forecast_it] 
            , axis = 0
        )

        sample_forecasts = np.expand_dims(
            np.split(sample_forecasts 
                    , len(target_path)
                    , axis=0
            )
            ,axis=0
        )

    
        sample_forecasts = np.concatenate(sample_forecasts, axis=0)#(ssm_num , bs , num_samples, 1)
        print('把预测结果保存在-->', forecast_result_saved_path)
        with open(forecast_result_saved_path , 'wb') as fp:
            pickle.dump(sample_forecasts , fp)
