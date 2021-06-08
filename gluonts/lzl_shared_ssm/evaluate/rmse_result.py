import numpy as np
import pandas as pd
import pickle
import heapq
import matplotlib.pyplot as plt
import os
if '/evaluate' in os.getcwd():
    os.chdir('../')
if '/lzl_shared_ssm' not in os.getcwd():
    os.chdir('gluonts/lzl_shared_ssm')

models = ['shared_ssm',
 'shared_ssm_no_sample',
 'shared_ssm_without_env' ,
  'prophet' ,'common_lstm',
   'arima','deep_state' ,'deepar',
    'DEEPAR', 'DEEPSTATE', 'PROPHET']
#('btc,eth' ,503)
#("UKX,VIX,SPX,SHSZ300,NKY" ,2867)
#("UKX,VIX,SPX,SHSZ300,NKY" , train, 2606)
past_length = 90
pred_length = 3
length = 2606
#length = 503 
slice_type = 'overlap'
target = 'UKX,VIX,SPX,SHSZ300,NKY'
#target = "btc,eth"
#! 记住在前后都加入下划线
init = "_xavier_"

# Shared ssm 专属的超参
dim_l = 4
K = 5
dim_u = 5
bs = 32;
lags = 7

#! 在这里修改 RMSE 的函数
# 维度分别是(ssm_num, bs, sq_len)
def calculate_accuracy(real, predict):
    real = np.array(real) + 1
    predict = np.array(predict) + 1
    res = np.sqrt(np.nanmean(np.square(real - predict))) / np.nanmean(np.abs(real))
    return res*100


eval_result_root = 'evaluate/results/{}_length({})_slice({})_past({})_pred({})'.format(
  target.replace(',' , '_') , length , slice_type, past_length , pred_length
)

print('current evaluate root path : ' , eval_result_root)

ground_truth_path = None;
for file in os.listdir(eval_result_root):
    if 'ground_truth' in file:
        ground_truth_path = file
        break
assert ground_truth_path != None,'当前当前并没有ground truth~~'


if __name__ == "__main__":

    # 这里对ground_truth进行合并,并且只保留两个部分，一个是start, target
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

    # 对模型产生的数据进行读取
    shared_ssm_result = []
    no_sample_acc_result = []
    deep_state_result = []
    shared_ssm_without_env_result = []
    common_lstm_result = []
    prophet_result  = []
    arima_result = []
    deepar_result = []
    DEEPAR_result = []
    DEEPSTATE_result = []
    PROPHET_result = []

    for path in os.listdir(eval_result_root):

        shared_ssm_con = ('shared_ssm' + init) in path
        shared_ssm_con_2 = 'num_samples' in path
        #超参选择
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
                # mse = np.nanmean(
                #     np.square(np.mean(single_result,axis=2)[:,:,-pred_length:] - ground_truth_target[:, :, -pred_length:]))
                # mse = np.nanmean(np.square(np.mean(single_result, 2) - ground_truth_target[:, :, -pred_length:]))
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
                no_sample_acc_result.append(acc)
                continue


        elif 'deepstate' in path:
            with open(os.path.join(eval_result_root, path), 'rb') as fp:
                single_result = pickle.load(fp)
                # mse = np.nanmean(np.square(np.mean(single_result, 2) - ground_truth_target[:, :, -pred_length:]))
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
                # mse = np.nanmean(
                #     np.square(single_result[:, :, -pred_length:] - ground_truth_target[:, :, -pred_length:]))
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
            # print('lstm result file name -->' , path)
            with open(os.path.join(eval_result_root , path) , 'rb') as fp:
                single_result = pickle.load(fp)
                single_result = single_result.squeeze(axis=-1)
                # mse = np.nanmean(
                #     np.square(single_result[:, :, -pred_length:] - ground_truth_target[:, :, -pred_length:]))
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
                # mse = np.nanmean(np.square(np.mean(single_result, 2) - ground_truth_target[:, :, -pred_length:]))
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
                # mse = np.nanmean(np.square(np.mean(single_result, 2) - ground_truth_target[:, :, -pred_length:]))
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
                # mse = np.nanmean(np.square(np.mean(single_result, 2) - ground_truth_target[:, :, -pred_length:]))
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
                # mse = np.nanmean(np.square(np.mean(single_result, 2) - ground_truth_target[:, :, -pred_length:]))
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
                # mse = np.nanmean(np.square(np.mean(single_result, 2) - ground_truth_target[:, :, -pred_length:]))
                acc = calculate_accuracy(
                    real=ground_truth_target[:, :, -pred_length:],
                    predict=np.mean(single_result, 2) if len(single_result.shape)==4 else single_result
                )
                acc = np.round(acc, 2)

                PROPHET_result.append(acc)
                continue

        else:
            # print('你可能命名有问题 -->' , path)
            pass




    acc_result = {}
    # top_num = 12
    for model in models:
        if model == 'shared_ssm':
            acc_result[model] = shared_ssm_result
        if model == 'shared_ssm_no_sample':
            acc_result[model] = no_sample_acc_result
        if model == 'deep_state':
            acc_result[model] =  deep_state_result
        if model  == 'shared_ssm_without_env' :
            acc_result[model] = shared_ssm_without_env_result
        if model == 'common_lstm':
            acc_result[model] = common_lstm_result
        if model == 'prophet':
            acc_result[model] = prophet_result
        if model == 'arima':
            acc_result[model] = arima_result
        if model == 'deepar':
            acc_result[model] = deepar_result
        if model == 'DEEPAR':
            acc_result[model] = DEEPAR_result
        if model == 'DEEPSTATE':
            acc_result[model] = DEEPSTATE_result
        if model == 'PROPHET':
            acc_result[model] = PROPHET_result
    print('---acc 指标结果 ---')
    for key, value in acc_result.items():
        print(key)
        for i in value:
            print("   ", i);
    print('---acc 指标完 ---')