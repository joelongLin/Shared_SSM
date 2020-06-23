import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
if '/evaluate' in os.getcwd():
    os.chdir('../')
if '/lzl_shared_ssm' not in os.getcwd():
    os.chdir('gluonts/lzl_shared_ssm')

models = ['shared_ssm', 'shared_ssm_without_env' , 'prophet' ,'common_lstm', 'common_lstm(time)' ,'deep_state']



common_lstm_result_path = 'common_lstm(btc,eth)_freq(1D)_past(90)_pred(5)_LSTM(2-40)_use_time(False)_loss_origin(True)_epoch(60)_bs(32)_bn(13)_lr(0.001)_dropout(0.1).pkl'
common_lstm_with_time_result_path = 'common_lstm(btc,eth)_freq(1D)_past(90)_pred(5)_LSTM(2-40)_use_time(True)_loss_origin(True)_epoch(60)_bs(32)_bn(13)_lr(0.001)_dropout(0.1).pkl'
shared_ssm_result_path = 'shared_ssm(btc,eth)_freq(1D)_lags(0)_past(90)_pred(5)_u(5)_l(4)_K(2)_T(2-40)_E(2-50)_α(50)_epoch(50)_bs(32)_bn(13)_lr(0.001)_initKF(0.05)_dropout(0.5)_5.pkl'
shared_ssm_without_env_result_path = 'shared_ssm_without_env(btc,eth)_freq(1D)_lags(0)_past(90)_pred(5)_u(5)_l(4)_K(2)_T(2-40)_E(2-50)_α(50)_epoch(50)_bs(32)_bn(13)_lr(0.001)_initKF(0.05)_dropout(0.5)_1.pkl'
prophet_result_path = 'prophet(btc,eth)_freq(1D)_lags(0)_past(90)_pred(5).pkl'
deep_state_result_path = 'deepstate(btc,eth)_freq(1D)_past(90)_pred(5)_LSTM(2-40)_trend(False)_epoch(50)_bs(32)_bn(13)_lr(0.001)_dropout(0.1)_4.pkl'
ground_truth_path = 'ground_truth(btc,eth)_freq(1D)_lags(0)_past(90)_pred(5)_u(5)_l(4)_K(2)_T(2-40)_E(2-50)_α(50)_epoch(50)_bs(32)_bn(13)_lr(0.001)_initKF(0.03)_dropout(0.5).pkl'

params = shared_ssm_result_path.split('_')
target = params[1][params[1].index('(')+1 : params[1].index(')')]
past_length = int(params[4][params[4].index('(')+1 : params[4].index(')')])
freq = params[2][params[2].index('(')+1 : params[2].index(')')]
pred_length = int(params[5][params[5].index('(')+1 : params[5].index(')')])
plot_length = pred_length*5
eval_result_root = 'evaluate/results/btc_eth_past({})_pred({})'.format(past_length, pred_length)
plot_pic_root = 'evaluate/pic/btc_eth_past({})_pred({})'.format(past_length , pred_length)

if __name__ == "__main__":

    # 这里对ground_truth进行合并,并且只保留两个部分，一个是start, target
    time_start = [] ; ground_truth_target = [];
    with open(os.path.join(eval_result_root , ground_truth_path), 'rb') as fp:
        ground_truth = pickle.load(fp)
    for batch in ground_truth:
        time_start += batch['start']
        batch_target = np.concatenate([batch['past_target'], batch['future_target']], axis=-2)
        ground_truth_target.append(batch_target)
    ground_truth_target = np.concatenate(ground_truth_target, axis=1).squeeze(axis=-1)
    ground_truth_target[ground_truth_target == 0] = np.NaN

    # 对模型产生的数据进行读取
    with open(os.path.join(eval_result_root , prophet_result_path), 'rb') as fp:
        prophet_result = pickle.load(fp)

    with open(os.path.join(eval_result_root , shared_ssm_result_path) , 'rb') as fp:
        shared_ssm_result = pickle.load(fp)
        shared_ssm_result = shared_ssm_result.squeeze(axis=-1)

    with open(os.path.join(eval_result_root , shared_ssm_without_env_result_path) , 'rb') as fp:
        shared_ssm_without_env_result = pickle.load(fp)
        shared_ssm_without_env_result = shared_ssm_without_env_result.squeeze(axis=-1)

    with open(os.path.join(eval_result_root , common_lstm_result_path) ,'rb') as fp:
        common_lstm_result = pickle.load(fp)
        common_lstm_result = common_lstm_result.squeeze(axis=-1)

    with open(os.path.join(eval_result_root , common_lstm_with_time_result_path), 'rb') as fp:
        common_lstm_result_with_time = pickle.load(fp)
        common_lstm_result_with_time = common_lstm_result_with_time.squeeze(axis=-1)

    with open(os.path.join(eval_result_root , deep_state_result_path) , 'rb') as fp:
        deep_state_result = pickle.load(fp)

    if not os.path.exists(plot_pic_root):
        os.makedirs(plot_pic_root)

    # for ssm_no in np.arange(0,ground_truth_target.shape[0]):
    #     # 为了获得正确的time数据
    #     time_start_no = 0
    #     for sample in np.arange(ground_truth_target.shape[1]):
    #         print('正在构建' , target.split(',')[ssm_no] ,'数据集的第', sample ,'个样本~')
    #         pic_name = os.path.join(plot_pic_root , 'dataset({})_past({})_pred({})_sample({})'.format(target.split(',')[ssm_no],past_length,pred_length,sample))
    #         #设置 X轴的数据
    #         all_time_range = pd.date_range(time_start[time_start_no], periods=ground_truth_target.shape[2], freq=freq)
    #         time_start_no += 1
    #         time_range = all_time_range[-plot_length:]
    #         pred_time_range = all_time_range[-pred_length:]
    #
    #         #GroundTruth
    #         s1 = ground_truth_target[ssm_no , sample ,-plot_length:]
    #
    #         # Models' Result
    #         s2 = shared_ssm_result[ssm_no , sample ,-pred_length:]
    #         s3 = shared_ssm_without_env_result[ssm_no , sample , -pred_length:]
    #         s4 = np.mean(prophet_result[ssm_no , sample ], axis=0)[-pred_length:]
    #         s5 = common_lstm_result[ssm_no , sample , -pred_length:]
    #         s6 = common_lstm_result_with_time[ssm_no , sample , -pred_length:]
    #         s7 = np.mean(deep_state_result[ssm_no, sample ],axis=0)[-pred_length:]
    #
    #         fig = plt.figure(figsize=(28, 21))
    #         ax = fig.add_subplot(1, 1, 1)
    #         ax.set_title('DATASET({}) TRAIN({}) PRED({})'.format(target.split(',')[ssm_no], past_length , pred_length), fontsize=30)
    #         ax.plot(time_range, s1, linestyle='-', color='tab:green', marker='D', label='Truth')
    #         ax.plot(pred_time_range, s2, linestyle='-.', color='tab:blue', marker='o', label='shared_ssm_pred_mean')
    #         ax.plot(pred_time_range , s3, linestyle='-.' , color = 'tab:cyan', marker='o' , label='shared_ssm(without_env)')
    #         ax.plot(pred_time_range, s4, linestyle='--', color='tab:purple', marker='v', label='prophet_sample_mean')
    #         ax.plot(pred_time_range, s5, linestyle='dotted', color='tab:orange', marker='*', label='common_lstm')
    #         ax.plot(pred_time_range, s6, linestyle='dotted', color='y', marker='P', label='common_lstm(time)')
    #         ax.plot(pred_time_range, s7, linestyle='--', color='tab:grey', marker='8', label='deep_state_sample_mean')
    #         ax.xaxis.set_tick_params(labelsize=21)
    #         ax.yaxis.set_tick_params(labelsize=21)
    #         ax.legend(prop={'size': 21}, edgecolor='red', facecolor='#e8dfdf')
    #         plt.savefig(pic_name)
    #         plt.close(fig)

    mse_result = {}
    for model in models:
        if model == 'shared_ssm':
            mse = np.nanmean(np.square(shared_ssm_result[:,:,-pred_length:] - ground_truth_target[:,:,-pred_length:]))
            mse_result[model] = mse
        if model == 'shared_ssm_without_env':
            mse = np.nanmean(np.square(shared_ssm_without_env_result[:, :, -pred_length:] - ground_truth_target[:, :, -pred_length:]))
            mse_result[model] = mse
        if model == 'prophet':
            mse = np.nanmean(np.square(np.mean(prophet_result, 2) - ground_truth_target[:,:,-pred_length:]))
            mse_result[model] = mse
        if model == 'common_lstm':
            mse = np.nanmean(np.square(common_lstm_result - ground_truth_target[:, :, -pred_length:]))
            mse_result[model] = mse
        if model == 'common_lstm(time)':
            mse = np.nanmean(np.square(common_lstm_result_with_time - ground_truth_target[:, :, -pred_length:]))
            mse_result[model] = mse
        if model == 'deep_state':
            mse = np.nanmean(np.square(np.mean(deep_state_result, 2) - ground_truth_target[:, :, -pred_length:]))
            mse_result[model] = mse

    print('---mse 指标结果 ---')
    for key, value in mse_result.items():
        print(key , '-->' , value)
    print('---mse 指标完 ---')
