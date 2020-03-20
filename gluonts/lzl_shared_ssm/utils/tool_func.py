from gluonts.dataset.util import to_pandas
import logging
import tensorflow as tf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gluonts.dataset.field_names import FieldName
import os
import numpy as np
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
TARGET_DIM = 4
def time_format_from_frequency_str(freq_str: str) :
    """
    根据freq str 返回合适的 time_stamp format
    Parameters
    ----------

    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
        B       business day frequency
        C       custom business day frequency (experimental)
        D       calendar day frequency
        W       weekly frequency
        M       month end frequency
        BM      business month end frequency
        CBM     custom business month end frequency
        MS      month start frequency
        BMS     business month start frequency
        CBMS    custom business month start frequency
        Q       quarter end frequency
        BQ      business quarter endfrequency
        QS      quarter start frequency
        BQS     business quarter start frequency
        A       year end frequency
        BA      business year end frequency
        AS      year start frequency
        BAS     business year start frequency
        BH      business hour frequency
        H       hourly frequency
        T, min  minutely frequency
        S       secondly frequency
        L, ms   milliseonds
        U, us   microseconds
        N       nanoseconds

    """

    features_by_offsets = {
        offsets.YearOffset: '%Y',
        offsets.MonthOffset: '%Y-%m',
        offsets.Week: '%Y-%W',
        offsets.Day: '%Y-%m-%d',
        offsets.BusinessDay: '%Y-%m-%d',
        offsets.Hour: '%Y-%m-%d %H',
        offsets.Minute: '%Y-%m-%d %H:%M',
    }

    offset = to_offset(freq_str)

    for offset_type, format_pattern in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return format_pattern


def create_dataset_if_not_exist(paths, start, past_length , pred_length , slice , timestep , freq):
    for name in paths:
        path = paths[name]
        if not os.path.exists(path):
            logging.info('there is no dataset [%s] , creating...' % name)
            os.system('python data_process/preprocessing.py -st={} -d={} -t={} -p={} -s={} -n={} -f={}'
                      .format( start
                              , name
                              , past_length
                              , pred_length
                              , slice
                              , timestep
                              , freq))
        else:
            logging.info(' dataset [%s] was found , good~~~' % name)

def weighted_average(
   metrics, weights = None, axis=None
):
    """
    计算加权平均
    metrics  #(ssm_num , bs,seq_length )
    weights  #(ssm_num , bs , seq_length)
    axis   metrics中需要加权的

    """

    if weights is not None:
        weighted_tensor = metrics * weights
        sum_weights = tf.math.maximum(1.0, tf.math.reduce_sum(weights,axis=axis))
        return tf.math.reduce_sum(weighted_tensor,axis=axis) / sum_weights
    else:
        return tf.math.reduce_mean(metrics, axis=axis)

def sample_without_put_back(samples, size):
    '''
    :param samples: 需要进行采样的集合
    :param size: 采样的大小
    :return: 不放回的采样
    '''

    result = []
    for i in range(size):
        z = np.random.choice(samples, 1)[0]
        result.append(z)
        index = np.where(samples == z)
        samples = np.delete(samples, index)
    return result

def plot_train_pred(path, data, pred, batch, epoch, plot_num, time_start, freq):
    '''
    :param path: 存放图片的地址
    :param data: 真实数据 #(ssm_num , bs , seq , 1)
    :param pred: 模型训练时预测的 每个时间步的 期望, 维度同 data
    :param batch: 辨明当前的
    :param epoch:当前的epoch
    :param plot_num: 当前情况下需要输出的图片数量
    :param time_start : 绘图的初始时间
    :param freq : 当前的时间间隔
    :return:
    '''
    #先把多余的维度去除
    data = np.squeeze(data , -1)
    pred = np.squeeze(pred ,-1)
    root_path = os.path.join(path , 'train_pred_pic')
    #当前采样
    samples_no = sample_without_put_back(np.arange(data.shape[1]) , plot_num)
    current_dir = os.path.join(root_path, 'epoch({})'.format(epoch))
    if not os.path.isdir(current_dir):
        os.makedirs(current_dir)
    for ssm_no in np.arange(data.shape[0]):
        for sample in samples_no:
            pic_name = os.path.join(current_dir , 'batch_no({})_ssm_no({})_sample({})'.format(batch,ssm_no,sample))
            time_range = pd.date_range(time_start[sample], periods=data.shape[2], freq=freq)
            s1 = data[ssm_no , sample]
            s2 = pred[ssm_no , sample]
            fig = plt.figure(figsize=(28, 21))

            ax = fig.add_subplot(1, 1, 1)
            ax.set_title('EPOCHE({} prediction result)'.format(epoch), fontsize=30)
            ax.plot(time_range, s1, linestyle='-', color='tab:green', marker='D', label='Truth')
            ax.plot(time_range, s2, linestyle='-.', color='tab:blue', marker='o', label='prediction')
            ax.xaxis.set_tick_params(labelsize=21)
            ax.yaxis.set_tick_params(labelsize=21)
            ax.legend(prop={'size': 31}, edgecolor='red', facecolor='#e8dfdf')
            plt.savefig(pic_name)
            plt.close(fig)

def plot_train_pred_NoSSMnum(path, data, pred, batch, epoch,ssm_no, plot_num, time_start, freq):
    '''
    与上述函数相比，不同的是，data以及pred都没有ssm_num维
    :param path: 存放图片的地址
    :param data: 真实数据 #( bs , seq , 1)
    :param pred: 模型训练时预测的 每个时间步的 期望, 维度同 data
    :param batch: 辨明当前的
    :param epoch:当前的epoch
    :param plot_num: 当前情况下需要输出的图片数量
    :param time_start : 绘图的初始时间
    :param freq : 当前的时间间隔
    :return:
    '''
    #先把多余的维度去除
    data = np.squeeze(data , -1) #(bs, seq)
    pred = np.squeeze(pred ,-1)
    root_path = os.path.join(path , 'train_pred_pic')
    #当前采样
    samples_no = np.random.choice(np.arange(data.shape[0]) , plot_num)
    current_dir = os.path.join(root_path, 'epoch({})'.format(epoch))
    if not os.path.isdir(current_dir):
        os.makedirs(current_dir)
    for sample in samples_no:
        pic_name = os.path.join(current_dir , 'batch_no({})_ssm_no({})_sample({})'.format(batch,ssm_no,sample))
        time_range = pd.date_range(time_start[sample], periods=data.shape[1], freq=freq)
        s1 = data[sample]
        s2 = pred[sample]
        fig = plt.figure(figsize = (28,21))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(time_range , s1 , linestyle = '-' , color = 'tab:green' , marker = 'D' ,label = 'Truth')
        ax.plot(time_range , s2 , linestyle = '-.' , color = 'tab:blue' , marker = 'o' , label = 'prediction')
        ax.xaxis.set_tick_params(labelsize=21)
        ax.yaxis.set_tick_params(labelsize=21)
        ax.legend(prop={'size': 31},edgecolor='red', facecolor='#e8dfdf')
        plt.savefig(pic_name)
        plt.close(fig)
    pass

def plot_train_epoch_loss(
        result:dict,
        path: str
):
    epoches = list(result.keys())
    loss = list(result.values())
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epoches, loss, color='tab:blue' , marker='D')
    ax.set(xlabel='epoch (s)', ylabel='filter negative log likelihood',
           title='epoch information when training')
    plt.savefig(os.path.join(path , 'train.png'))
    plt.close(fig)

def complete_batch(batch,batch_size):
    # 对每一个 key都进行处理
    if len(batch[FieldName.START]) == batch_size:
        return batch , batch_size
    for attr in batch.keys():
        batch_value = batch[attr]
        if isinstance(batch_value , list):
            batch_num = len(batch_value)
            batch[attr] = batch_value + [batch_value[-1]]*(batch_size-batch_num)
        elif isinstance(batch_value, np.ndarray):
            #表示所有非四维的
            if len(batch_value.shape) != TARGET_DIM:
                batch_num = batch_value.shape[0]
                complete_shape = [batch_size-batch_num]+list(batch_value.shape[1:])
                batch[attr] = np.concatenate([batch_value , np.zeros(shape=complete_shape)], axis=0)
            else:
                batch_num = batch_value.shape[1]
                complete_shape = [batch_value.shape[0],batch_size - batch_num] + list(batch_value.shape[2:])
                batch[attr] = np.concatenate([batch_value, np.zeros(shape=complete_shape)], axis=1)
    return batch ,batch_num


def del_previous_model_params(path):
    '''
    :param path:  该组参数组的主目录
    :return:  删除之前epoch的参数文件
    '''
    path = os.path.join(path,'model_params')
    if not os.path.isdir(path):
        return
    for files in os.listdir(path):
        if files.endswith(".data-00000-of-00001") \
                or files.endswith(".index") \
                or files.endswith(".meta"):
            os.remove(os.path.join(path,files))


# 画原有序列
def plot_original(train_entry , test_entry , no):
    test_series = to_pandas(test_entry)
    train_series = to_pandas(train_entry)

    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 7))

    train_series.plot(ax=ax[0])
    ax[0].grid(which="both")
    ax[0].legend(["train series"], loc="upper left")

    test_series.plot(ax=ax[1])
    ax[1].axvline(train_series.index[-1], color='r')  # end of train dataset
    ax[1].grid(which="both")
    ax[1].legend(["test series", "end of train series"], loc="upper left")

    plt.savefig('pic/original_entry_{}.png'.format(no))