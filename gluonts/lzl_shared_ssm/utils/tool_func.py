import tensorflow as tf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np

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
    ssm_no = np.random.choice(np.arange(data.shape[0]) , 1)[0] # 如果不进行slice会导致后面值多一维度
    samples_no = np.random.choice(np.arange(data.shape[1]) , plot_num)
    current_dir = os.path.join(root_path, 'epoch({})'.format(epoch))
    if not os.path.isdir(current_dir):
        os.makedirs(current_dir)
    for sample in samples_no:
        pic_name = os.path.join(current_dir , 'batch_no({})_ssm_no({})_sample({})'.format(batch,ssm_no,sample))
        time_range = pd.date_range(time_start[sample], periods=data.shape[2], freq=freq)
        s1 = data[ssm_no , sample]
        s2 = pred[ssm_no , sample]
        fig = plt.figure(figsize=(28, 21))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(time_range, s1, linestyle='-', color='tab:green', marker='D', label='Truth')
        ax.plot(time_range, s2, linestyle='-.', color='tab:blue', marker='o', label='prediction')
        ax.xaxis.set_tick_params(labelsize=21)
        ax.yaxis.set_tick_params(labelsize=21)
        ax.legend(prop={'size': 31}, edgecolor='red', facecolor='#e8dfdf')
        plt.savefig(pic_name)
        plt.close(fig)
    pass

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