import numpy as np
import pandas as pd
import pickle
import heapq
import matplotlib.pyplot as plt
import os
import sys
if '/evaluate' in os.getcwd():
    os.chdir('../')
if '/lzl_shared_ssm' not in os.getcwd():
    os.chdir('gluonts/lzl_shared_ssm')

models = ['shared_ssm',
 'shared_ssm_no_sample',
 'shared_ssm_without_env' ,
  'prophet' ,'common_lstm'
  ,'deep_state' ,'deepar'
  ,'DEEPAR', 'DEEPSTATE', 'PROPHET']
#('btc,eth' ,503)
#("UKX,VIX,SPX,SHSZ300,NKY" ,2606)
past_length = 60
pred_length = 3
slice_type = 'overlap'

#length = 1168
#length = 2606
length = 503
#target = 'PM25,PM10,NO2'
# target = 'UKX,VIX,SPX,SHSZ300,NKY'
target = "btc,eth"

# SSSM parameter
dim_l = 4
K = 3;
dim_u = 5
bs = 32;
lags = 3;

check_noise = ""
if(len(sys.argv) >= 2):
    check_noise = sys.argv[1]
eval_result_root = 'evaluate/results{}/{}_length({})_slice({})_past({})_pred({})'.format(
  "" if check_noise=="" else "_" + check_noise, target.replace(',' , '_') , length , slice_type, past_length , pred_length
)


def calculate_accuracy(real, predict):
    real = np.array(real) + 1
    predict = np.array(predict) + 1
    percentage = 1 - np.sqrt(np.nanmean(np.square((real - predict) / real)))
    return percentage * 100

print('current evaluate root path : ' , eval_result_root)

ground_truth_path = None;
for file in os.listdir(eval_result_root):
    if 'ground_truth' in file:
        ground_truth_path = file
        break
assert ground_truth_path != None,'NO ground truth~~'


if __name__ == "__main__":

    time_start = [] ; ground_truth_target = [];
    with open(os.path.join(eval_result_root , ground_truth_path), 'rb') as fp:
        ground_truth = pickle.load(fp)
    for batch in ground_truth:
        time_start += batch['start']
        batch_target = np.concatenate([batch['past_target'], batch['future_target']], axis=-2)
        ground_truth_target.append(batch_target)
    
    # (ssm_num , series , past+pred)
    ground_truth_target = np.concatenate(ground_truth_target, axis=1).squeeze(axis=-1)
    ground_truth_target[ground_truth_target == 0] = np.NaN

    # RESULT OF MODEL
    shared_ssm_result = []
    shared_ssm_no_sample = []
    deep_state_result = []
    shared_ssm_without_env_result = []
    common_lstm_result = []
    prophet_result  = []
    deepar_result = []
    DEEPAR_result = []
    DEEPSTATE_result = []
    PROPHET_result = []

    for path in sorted(os.listdir(eval_result_root)):

        shared_ssm_con = 'shared_ssm' in path
        shared_ssm_con_2 = 'num_samples' in path
        # SSSM hyperparameter 
        shared_ssm_con_3 = 'lags({})'.format(lags) in path \
        and 'u({})'.format(dim_u) in path \
        and 'l({})'.format(dim_l) in path \
        and 'bs({})'.format(bs) in path \
        and 'K({})'.format(K) in path
        
        if shared_ssm_con and shared_ssm_con_2 and shared_ssm_con_3 :
            print('shared ssm sample with mean and covariance -->' , path)
            with open(os.path.join(eval_result_root , path) , 'rb') as fp:
                single_result = pickle.load(fp)#(ssm , series ,samples, pred, 1 )
                single_result = single_result.squeeze(axis=-1)
                acc = calculate_accuracy(
                    real=ground_truth_target[:, :, -pred_length:],
                    predict=np.mean(single_result, axis=2)[:, :, -pred_length:]
                )
                acc = np.round(acc ,2 )
                shared_ssm_result.append(acc)
                continue

        elif shared_ssm_con and shared_ssm_con_3 :
            print('shared ssm use mean as output --->' , path)
            with open(os.path.join(eval_result_root , path) , 'rb') as fp:
                single_result = pickle.load(fp)#(ssm , series , pred, 1 )
                single_result = single_result.squeeze(axis=-1)
                acc = calculate_accuracy(
                    real=ground_truth_target[:, :, -pred_length:],
                    predict=single_result[:, :, -pred_length:]
                )
                acc = np.round(acc, 2)
                shared_ssm_no_sample.append(acc)
                continue


        elif 'deepstate' in path:
            with open(os.path.join(eval_result_root, path), 'rb') as fp:
                single_result = pickle.load(fp)
                acc = calculate_accuracy(
                    real=ground_truth_target[:, :, -pred_length:],
                    predict=np.mean(single_result, 2)
                )
                acc = np.round(acc, 2)
                deep_state_result.append(acc)
                continue

        elif 'without_env' in path and shared_ssm_con_3:
            with open(os.path.join(eval_result_root , path) , 'rb') as fp:
                single_result = pickle.load(fp)
                single_result = single_result.squeeze(axis=-1)
                if len(single_result.shape) == 3:
                    acc = calculate_accuracy(
                        real=ground_truth_target[:, :, -pred_length:],
                        predict=single_result[:, :, -pred_length:]
                    )
                elif len(single_result.shape) == 4 :
                    acc = calculate_accuracy(
                        real=ground_truth_target[:, :, -pred_length:],
                        predict=np.mean(single_result, axis=2)[:, :, -pred_length:]
                    )
                else:
                    raise RuntimeError("shared ssm without env 的output不合法");
                acc = np.round(acc, 2)
                shared_ssm_without_env_result.append(acc)
                continue

        elif 'common_lstm' in path:
            with open(os.path.join(eval_result_root , path) , 'rb') as fp:
                single_result = pickle.load(fp)
                single_result = single_result.squeeze(axis=-1)
        
                acc = calculate_accuracy(
                    real=ground_truth_target[:, :, -pred_length:],
                    predict=single_result[:, :, -pred_length:]
                )
                acc = np.round(acc, 2)
                common_lstm_result.append(acc)
                continue

        elif 'prophet' in path:
            with open(os.path.join(eval_result_root, path), 'rb') as fp:
                single_result = pickle.load(fp)
                
                acc = calculate_accuracy(
                    real=ground_truth_target[:, :, -pred_length:],
                    predict=np.mean(single_result, 2)
                )
                acc = np.round(acc, 2)

                prophet_result.append(acc)
                continue
        
        
        elif 'deepar' in path:
            with open(os.path.join(eval_result_root, path), 'rb') as fp:
                single_result = pickle.load(fp)
                
                acc = calculate_accuracy(
                    real=ground_truth_target[:, :, -pred_length:],
                    predict=np.mean(single_result, 2) if len(single_result.shape)==4 else single_result
                )
                acc = np.round(acc, 2)

                deepar_result.append(acc)
                continue
        
        elif 'DEEPAR' in path:
            with open(os.path.join(eval_result_root, path), 'rb') as fp:
                if "lag({})".format(lags) not in path:
                    continue;
                single_result = pickle.load(fp)
                
                acc = calculate_accuracy(
                    real=ground_truth_target[:, :, -pred_length:],
                    predict=np.mean(single_result, 2) if len(single_result.shape)==4 else single_result
                )
                acc = np.round(acc, 2)

                DEEPAR_result.append(acc)
                continue
        
        elif 'DEEPSTATE' in path:
            with open(os.path.join(eval_result_root, path), 'rb') as fp:
                if "lag({})".format(lags) not in path:
                    continue;
                single_result = pickle.load(fp)
                
                acc = calculate_accuracy(
                    real=ground_truth_target[:, :, -pred_length:],
                    predict=np.mean(single_result, 2) if len(single_result.shape)==4 else single_result
                )
                acc = np.round(acc, 2)

                DEEPSTATE_result.append(acc)
                continue
        
        elif 'PROPHET' in path:
            with open(os.path.join(eval_result_root, path), 'rb') as fp:
                if "lag({})".format(lags) not in path:
                    continue;
                single_result = pickle.load(fp)
                
                acc = calculate_accuracy(
                    real=ground_truth_target[:, :, -pred_length:],
                    predict=np.mean(single_result, 2) if len(single_result.shape)==4 else single_result
                )
                acc = np.round(acc, 2)

                PROPHET_result.append(acc)
                continue
        
    


        else:
            print('Invalid output -->' , path)
            pass




    acc_result = {}
    # top_num = 12
    for model in models:
        if model == 'shared_ssm':
            acc_result[model] = shared_ssm_result
        if model == 'shared_ssm_no_sample':
            acc_result[model] = shared_ssm_no_sample
        if model == 'deep_state':
            acc_result[model] =  deep_state_result
        if model  == 'shared_ssm_without_env' :
            acc_result[model] = shared_ssm_without_env_result
        if model == 'common_lstm':
            acc_result[model] = common_lstm_result
        if model == 'prophet':
            acc_result[model] = prophet_result
        if model == 'deepar':
            acc_result[model] = deepar_result
        if model == 'DEEPAR':
            acc_result[model] = DEEPAR_result
        if model == 'DEEPSTATE':
            acc_result[model] = DEEPSTATE_result
        if model == 'PROPHET':
            acc_result[model] = PROPHET_result
    print('---acc START ---')
    for key, value in acc_result.items():
        print(key)
        for i in value:
            print("   ", i);
    print('---acc END ---')
