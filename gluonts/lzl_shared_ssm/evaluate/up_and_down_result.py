import numpy as np
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt
import os
if '/evaluate' in os.getcwd():
    os.chdir('../')
if '/lzl_shared_ssm' not in os.getcwd():
    os.chdir('gluonts/lzl_shared_ssm')

models = ['shared_ssm', 'shared_ssm_without_env' , 'prophet' ,'common_lstm', 'common_lstm(time)' ,'deep_state']

# shared_ssm_without_env_result_path = 'shared_ssm_without_env(btc,eth)_freq(1D)_lags(0)_past(30)_pred(1)_u(5)_l(4)_K(2)_T(2-40)_E(2-50)_α(50)_epoch(50)_bs(32)_bn(15)_lr(0.001)_initKF(0.05)_dropout(0.5).pkl'
# shared_ssm_result_path = 'shared_ssm(btc,eth)_freq(1D)_lags(0)_past(30)_pred(1)_u(5)_l(4)_K(2)_T(2-40)_E(2-50)_α(50)_epoch(50)_bs(32)_bn(15)_lr(0.001)_initKF(0.05)_dropout(0.5).pkl'
# deep_state_result_path = 'deepstate(btc,eth)_freq(1D)_past(30)_pred(1)_LSTM(2-40)_trend(False)_epoch(50)_bs(32)_bn(15)_lr(0.001)_dropout(0.5).pkl'
# common_lstm_result_path = 'common_lstm(btc,eth)_freq(1D)_past(30)_pred(1)_LSTM(2-40)_use_time(False)_loss_origin(True)_epoch(50)_bs(32)_bn(15)_lr(0.001)_dropout(0.5).pkl'
# prophet_result_path = 'prophet(btc,eth)_freq(1D)_lags(0)_past(30)_pred(1).pkl'
# ground_truth_path = 'ground_truth(btc,eth)_start(2018-08-02)_freq(1D)_past(30)_pred(1).pkl'

# params = ground_truth_path.split('_')
# target = params[1][params[1].index('(')+1 : params[1].index(')')]
# freq = params[3][params[3].index('(')+1 : params[3].index(')')]
# past_length = int(params[4][params[4].index('(')+1 : params[4].index(')')])
# pred_length = int(params[5][params[5].index('(')+1 : params[5].index(')')])
past_length = 30
slice = 'nolap'
pred_length = 1
eval_result_root = 'evaluate/results/btc_diff_eth_diff_slice({})_past({})_pred({})'.format(slice ,past_length , pred_length)
ground_truth_path = None;
for file in os.listdir(eval_result_root):
    if 'ground_truth' in file:
        ground_truth_path = file
        break
assert ground_truth_path != None,'当前当前并没有ground truth~~'

if __name__ == "__main__":

    # 这里对ground_truth进行合并,并且只保留两个部分，一个是start, target
    ground_truth_target = [];
    with open(os.path.join(eval_result_root , ground_truth_path), 'rb') as fp:
        ground_truth = pickle.load(fp)
    for batch in ground_truth:
        batch_target = np.concatenate([batch['past_target'], batch['future_target']], axis=-2)
        ground_truth_target.append(batch_target)
    ground_truth_target = np.concatenate(ground_truth_target, axis=1).squeeze(axis=-1)
    ground_truth_target[ground_truth_target == 0] = np.NaN

    #如果涨了，那么表示为1 ，如果跌了那么表示为0
    diff = np.reshape(ground_truth_target[:,:,-1],(-1) ) #(ssm , bs)
    filter_index = np.where(np.isnan(diff) != True)
    diff_filter = diff[filter_index]
    up_and_down = np.where(diff_filter>0 ,1 , 0)
    print('------全1的情况-------')
    print('accuracy : ',metrics.accuracy_score(up_and_down , np.ones(up_and_down.shape)))
    print('recall : ',metrics.recall_score(up_and_down, np.ones(up_and_down.shape)))
    print('macro-F1 : ',metrics.f1_score(up_and_down, np.ones(up_and_down.shape),average='macro'))
    print('micro-F1 : ',metrics.f1_score(up_and_down, np.ones(up_and_down.shape),average='micro'))
    print('-------end----------\n\n')

    print('------全0的情况-------')
    print('accuracy : ', metrics.accuracy_score(up_and_down, np.zeros(up_and_down.shape)))
    print('recall : ', metrics.recall_score(up_and_down, np.zeros(up_and_down.shape)))
    print('macro-F1 : ', metrics.f1_score(up_and_down, np.zeros(up_and_down.shape), average='macro'))
    print('micro-F1 : ', metrics.f1_score(up_and_down, np.zeros(up_and_down.shape), average='micro'))
    print('-------end----------\n\n')


    # 对模型产生的数据进行读取
    shared_ssm_result = {'accuracy':[] , 'recall':[] , 'macro-F1':[] ,'micro-F1':[]}
    deep_state_result = {'accuracy':[] , 'recall':[] , 'macro-F1':[] ,'micro-F1':[]}
    shared_ssm_without_env_result = {'accuracy':[] , 'recall':[] , 'macro-F1':[] ,'micro-F1':[]}
    common_lstm_result = {'accuracy':[] , 'recall':[] , 'macro-F1':[] ,'micro-F1':[]}
    prophet_result  = {'accuracy':[] , 'recall':[] , 'macro-F1':[] ,'micro-F1':[]}
    for path in os.listdir(eval_result_root):
        if 'shared_ssm(' in path  and 'lags(4)' in path:
            with open(os.path.join(eval_result_root , path) , 'rb') as fp:
                single_result = pickle.load(fp)
                single_result = single_result.squeeze(axis=-1)
                single_diff = np.reshape(single_result[:,:,-1]  ,(-1))[filter_index]  #ssm_num*bs
                model_up_and_down = np.where(single_diff>0 , 1, 0)
                shared_ssm_result['accuracy'].append(metrics.accuracy_score(
                    y_true=up_and_down , y_pred=model_up_and_down
                ))
                shared_ssm_result['recall'].append(metrics.recall_score(
                    y_true=up_and_down, y_pred=model_up_and_down
                ))
                shared_ssm_result['macro-F1'].append(metrics.f1_score(
                    y_true=up_and_down, y_pred=model_up_and_down,
                    average='macro'
                ))
                shared_ssm_result['micro-F1'].append(metrics.f1_score(
                    y_true=up_and_down, y_pred=model_up_and_down,
                    average='micro'
                ))
                continue

        if 'deepstate' in path:
            with open(os.path.join(eval_result_root, path), 'rb') as fp:
                deepstate_single_result = pickle.load(fp)
                single_diff = np.mean(deepstate_single_result, 2)[:, :, -1]
                single_diff = np.reshape(single_diff, (-1))[filter_index]

                model_up_and_down = np.where(single_diff > 0, 1, 0)
                deep_state_result['accuracy'].append(metrics.accuracy_score(
                    y_true=up_and_down, y_pred=model_up_and_down
                ))
                deep_state_result['recall'].append(metrics.recall_score(
                    y_true=up_and_down, y_pred=model_up_and_down
                ))
                deep_state_result['macro-F1'].append(metrics.f1_score(
                    y_true=up_and_down, y_pred=model_up_and_down,
                    average='macro'
                ))
                deep_state_result['micro-F1'].append(metrics.f1_score(
                    y_true=up_and_down, y_pred=model_up_and_down,
                    average='micro'
                ))
                continue

        if 'shared_ssm_without_env' in path:
            with open(os.path.join(eval_result_root , path) , 'rb') as fp:
                single_result = pickle.load(fp)
                single_result = single_result.squeeze(axis=-1)
                single_diff = np.reshape(single_result[:, :, -1] , (-1))[
                    filter_index]  # ssm_num*bs
                model_up_and_down = np.where(single_diff > 0, 1, 0)
                shared_ssm_without_env_result['accuracy'].append(metrics.accuracy_score(
                    y_true=up_and_down, y_pred=model_up_and_down
                ))
                shared_ssm_without_env_result['recall'].append(metrics.recall_score(
                    y_true=up_and_down, y_pred=model_up_and_down
                ))
                shared_ssm_without_env_result['macro-F1'].append(metrics.f1_score(
                    y_true=up_and_down, y_pred=model_up_and_down,
                    average='macro'
                ))
                shared_ssm_without_env_result['micro-F1'].append(metrics.f1_score(
                    y_true=up_and_down, y_pred=model_up_and_down,
                    average='micro'
                ))
                continue

        if 'common_lstm' in path:
            with open(os.path.join(eval_result_root , path) , 'rb') as fp:
                single_result = pickle.load(fp)
                single_result = single_result.squeeze(axis=-1)
                single_diff = np.reshape(single_result[:, :, -1] , (-1))[
                    filter_index]  # ssm_num*bs
                model_up_and_down = np.where(single_diff > 0, 1, 0)

                common_lstm_result['accuracy'].append(metrics.accuracy_score(
                    y_true=up_and_down, y_pred=model_up_and_down
                ))
                common_lstm_result['recall'].append(metrics.recall_score(
                    y_true=up_and_down, y_pred=model_up_and_down
                ))
                common_lstm_result['macro-F1'].append(metrics.f1_score(
                    y_true=up_and_down, y_pred=model_up_and_down,
                    average='macro'
                ))
                common_lstm_result['micro-F1'].append(metrics.f1_score(
                    y_true=up_and_down, y_pred=model_up_and_down,
                    average='micro'
                ))
                continue

        if 'prophet' in path:
            with open(os.path.join(eval_result_root, path), 'rb') as fp:
                deepstate_single_result = pickle.load(fp)
                single_diff = np.mean(deepstate_single_result, 2)[:, :, -1]
                single_diff = np.reshape(single_diff , (-1))[filter_index]

                model_up_and_down = np.where(single_diff > 0, 1, 0)
                prophet_result['accuracy'].append(metrics.accuracy_score(
                    y_true=up_and_down, y_pred=model_up_and_down
                ))
                prophet_result['recall'].append(metrics.recall_score(
                    y_true=up_and_down, y_pred=model_up_and_down
                ))
                prophet_result['macro-F1'].append(metrics.f1_score(
                    y_true=up_and_down, y_pred=model_up_and_down,
                    average='macro'
                ))
                prophet_result['micro-F1'].append(metrics.f1_score(
                    y_true=up_and_down, y_pred=model_up_and_down,
                    average='micro'
                ))
                continue




    up_and_down_metrics = {}
    for model in models:
        if model == 'shared_ssm':
            up_and_down_metrics[model] = shared_ssm_result
        if model == 'deep_state':
            up_and_down_metrics[model] = deep_state_result
        if model  == 'shared_ssm_without_env' :
            up_and_down_metrics[model] = shared_ssm_without_env_result
        if model == 'common_lstm':
            up_and_down_metrics[model] = common_lstm_result
        if model == 'prophet':
            up_and_down_metrics[model] = prophet_result

    for model_name, value in up_and_down_metrics.items():
        print('---%s 指标结果 ---'%(model_name))
        # for metric , metric_value in value.items():
        #     print(metric ,'\n' , np.mean(metric_value))
        print('macro-F1  \n' ,[np.round(v , 4) for v in value['macro-F1']])
        print('---%s 指标完 ---\n\n'%(model_name))