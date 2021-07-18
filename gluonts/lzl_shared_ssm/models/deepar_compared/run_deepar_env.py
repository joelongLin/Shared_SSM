import pandas as pd
import os
import sys

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
parser.add_argument('-e' ,'--env' , type=str , help='enviroment as dynamic_feat_real', default='gold')
parser.add_argument('-l' , '--lags' , type=int, help='featu_dynamic_real and target series lag', default=5)
parser.add_argument('-st','--start', type=str, help='start time of the dataset',default='2018-08-02')
parser.add_argument('-f'  ,'--freq' , type = str , help = 'frequency of dataset' , default='1D')
parser.add_argument('-T','--timestep', type=int, help='length of the dataset', default=503)
parser.add_argument('-pr'  ,'--pred' , type = int , help = 'length of prediction range' , default=5)
parser.add_argument('-pa'  ,'--past' , type = int , help = 'length of training range' , default=90)
parser.add_argument('-s'  ,'--slice' , type = str , help = 'type of sliding dataset' , default='overlap')
parser.add_argument('-bs'  ,'--batch_size' , type = int , help = 'batch size' , default=32)
parser.add_argument('-bn'  ,'--batch_num' , type = int , help = 'batch number per epoch' , default=13)
parser.add_argument('-ep'  ,'--epochs' , type = int , help = 'epochs to train' , default=1)
args = parser.parse_args()

item = args.target
start = args.start
# change "_"  in start
if("_" in start):
    start = start.replace("_", " ")
freq = args.freq
timestep = args.timestep
pred = args.pred
past = args.past
slice_style = args.slice
batch_size = args.batch_size
batch_num = args.batch_num
epochs = args.epochs
env = args.env
lags = args.lags
ds_name_prefix = 'data_process/processed_data/{}_{}_{}_{}.pkl'

result_params = '_'.join([
                                'freq(%s)'%(freq),
                               'past(%d)'%(past)
                                ,'pred(%d)'%(pred)
                                ,'env(%s)'%(env.replace(',' , '_'))
                                ,'lag(%d)'%(lags)
                             ]
                        )


if __name__ == '__main__':
    # load target 
    if slice_style == 'overlap':
        series = timestep - past - pred + 1
        print('Series num in each dataset: ', series)
    elif slice_style == 'nolap':
        series = timestep // (past + pred)
        print('Series num in each dataset: ', series)
    else:
        series = 1
        print('Series num in each dataset: ', series ,', is single series')
    result_root_path  = 'evaluate/results/{}_length({})_slice({})_past({})_pred({})'.format(item.replace(',' ,'_'), timestep , slice_style ,past , pred)
    if not os.path.exists(result_root_path):
        os.makedirs(result_root_path)
    
    forecast_result_saved_path = os.path.join(result_root_path,'DEEPAR_'  + result_params + '.pkl')
    forecast_result_saved_path = add_time_mark_to_file(forecast_result_saved_path)

    # target series path
    target_path = {ds_name: ds_name_prefix.format(
        '%s_start(%s)_freq(%s)' % (ds_name, start,freq), '%s_DsSeries_%d' % (slice_style, series),
        'train_%d' % past, 'pred_%d_dynamic_feat_real(%s)_lag(%d)' % (pred , env.replace(',', '_'), lags)
    ) for ds_name in item.split(',')}

    for name in target_path:
        path = target_path[name]
        if not os.path.exists(path):
            print(name, "creating...")
            print(os.getcwd())
            command = "python data_process/preprocessing_env.py --start {} --dataset {} --train_length {} --pred_length {} --slice {} --num_time_steps {} --freq {} --env {} --lags {}".format( args.start, name, past, pred, slice_style, timestep, freq, env, lags)
            print(command)
            os.system(command)
        else:
            print(name , "Dataset exists~~")
    if not os.path.exists(forecast_result_saved_path):
        # dimension of each target series should be 1, to check the num of ssm.
        target_ds = None;
        for item_name in target_path:
            item = target_path[item_name]
            print('import data : {}'.format(item))
            with open(item, 'rb') as fp:
                item_ds = pickle.load(fp)
                assert item_ds.metadata['dim'] == 1, 'dimension of target series should be 1'
                # merging dataset
                if target_ds == None:
                    target_ds = item_ds.test
                else:
                    target_ds.list_data += item_ds.test.list_data
                
        deepar_estimator = DeepAREstimator(
            freq=freq, 
            prediction_length=pred,
            context_length=past,
            lags_seq=[1],
            use_feat_dynamic_real = True,
            trainer = Trainer(
                epochs=epochs,
                batch_size=batch_size*len(target_path),
                num_batches_per_epoch=batch_num,
                learning_rate=0.001
            )
        )
        
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
        print('Result of prediction are stored in-->', forecast_result_saved_path)
        with open(forecast_result_saved_path , 'wb') as fp:
            pickle.dump(sample_forecasts , fp)
