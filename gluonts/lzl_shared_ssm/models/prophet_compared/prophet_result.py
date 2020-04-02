import matplotlib
matplotlib.use('Agg')
import os
os.chdir('../..')
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm

ds_name = 'eth'
maxlags=0
start = '2018-08-02'
freq = '1D'
timestep = 503
pred = 5
past = 498
plot_length = 5*pred
slice = ''
name_prefix = 'data_process/processed_data/{}_{}_{}_{}.pkl'
quantiles = [0.05 , 0.25 , 0.5 , 0.75 , 0.95]
save_path = 'logs/btc_eth(prophet)/pred_pic'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def alpha_for_percentile(p):
    return (p / 100.0) ** 0.3

if __name__ == '__main__':

    if slice == 'overlap':
        series = timestep - past - pred + 1
        print('每个数据集的序列数量为 ,', series)
    elif slice == 'nolap':
        series = timestep // (past + pred)
        print('每个数据集的序列数量为 ,', series)
    else:
        series = 1

    path = name_prefix.format('%s_start(%s)' % (ds_name, start), '%s_DsSeries_%d' % (slice, series),
        'train_%d' % past, 'pred_%d' % pred)
    with open(path , 'rb') as fp:
        groundtruth = pickle.load(fp)
    with open('models/prophet_compared/result/{}_series({})_sample_forecasts.pkl'.format(ds_name , series) ,'rb') as fp:
        forecasts = pickle.load(fp)

    for i in tqdm(range(series)):
        # 画图的主要线段
        all_time_range = pd.date_range(start=groundtruth.metadata['start'][i],periods=past+pred,freq=freq)[-plot_length:]
        pred_time_range =pd.date_range(start=forecasts[i].start_date, periods=forecasts[i].samples.shape[1] ,freq= forecasts[i].freq)[-plot_length:]
        s1 = np.squeeze(groundtruth.test.list_data[i]['target'])[-plot_length:]
        # nan_idx = np.isfinite(s1) == False
        # s1[nan_idx] = np.NaN
        s2 = forecasts[i].mean[-plot_length:]

        #进行中间补充，涂色
        ps_data = [forecasts[i].quantile(q) for q in quantiles]

        pic_name = os.path.join(save_path, 'Series({})_sample_no({})'.format(ds_name, i))
        fig = plt.figure(figsize=(28, 21))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('SERIES({} prediction result)'.format(ds_name), fontsize=30)
        ax.plot(all_time_range, s1, linestyle='-', color='tab:green', marker='D', label='Truth')
        ax.plot(pred_time_range, s2, linestyle='-.', color='tab:blue', marker='o', label='samples_mean')
        for j in range(len(ps_data) // 2):
            ax.fill_between(x=pred_time_range , y1=ps_data[j]
                            ,y2=ps_data[-j-1], facecolor='b' , interpolate=True
                            ,alpha=alpha_for_percentile(quantiles[j]))

        ax.xaxis.set_tick_params(labelsize=21)
        ax.yaxis.set_tick_params(labelsize=21)
        ax.legend(prop={'size': 31}, edgecolor='red', facecolor='#e8dfdf')
        plt.savefig(pic_name)
        plt.close(fig)