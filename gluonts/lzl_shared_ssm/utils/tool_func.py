from gluonts.dataset.util import to_pandas
import logging
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
        offsets.Hour: '%Y-%m-%d %H:%M:%S',
        offsets.Minute: '%Y-%m-%d %H:%M:%S',
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
            os.system('python data_process/preprocessing.py -st="{}" -d="{}" -t="{}" -p="{}" -s="{}" -n="{}" -f="{}"'
                      .format( start
                              , name
                              , past_length
                              , pred_length
                              , slice
                              , timestep
                              , freq))
        else:
            logging.info(' dataset [%s] was found , good~~~' % name)

def add_time_mark_to_file(path):
    '''
    给重复的文件名添加 time 字段
    :param path: 路径
    :return: 返回新的路径名
    '''
    count = 1
    if not os.path.exists(path):
        return path
    file_name_list = os.path.splitext(os.path.basename(path))
    father = os.path.split(path)[0]

    new_path = os.path.join(father ,file_name_list[0]+'_%d'%(count)+file_name_list[1])
    while os.path.exists(new_path):
        count += 1
        new_path = os.path.join(father, file_name_list[0] + '_%d'%(count) + file_name_list[1])

    return new_path

def add_time_mark_to_dir(path):
    '''
    给已经存在的文件夹添加 time 字段
    :param path: 路径
    :return: 新的文件夹路径
    '''
    if not os.path.exists(path):
        return path
    count = 1;
    father = os.path.split(path)[0]
    dir_name = os.path.split(path)[-1]

    new_path = os.path.join(father ,dir_name+'_%d'%(count))
    while os.path.exists(new_path):
        count += 1
        new_path = os.path.join(father ,dir_name+'_%d'%(count))

    return new_path

def weighted_average(
   metrics, weights = None, axis=None
):
    """
    计算加权平均
    metrics  #(ssm_num , bs,seq_length )
    weights  #(ssm_num , bs , seq_length)
    axis   metrics中需要加权的
    """
    import tensorflow as tf
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

def plot_train_pred(path, targets_name, data, pred, batch, epoch, plot_num, plot_length, time_start, freq, nan_data=0):
    '''
    :param path: 存放图片的地址
    :param targets_name: 预测数据集的名称
    :param data: 真实数据 #(ssm_num , bs , seq , 1)
    :param nan_data 缺失值的代表值，0
    :param pred: 模型训练时预测的 每个时间步的 期望, 维度同 data
    :param batch: 辨明当前的
    :param epoch:当前的epoch
    :param plot_num: 当前情况下需要输出的图片数量
    :param plot_length: 当前序列画图的长度
    :param time_start : 绘图的初始时间
    :param freq : 当前的时间间隔
    :return:
    '''
    #先把多余的维度去除
    data = np.squeeze(data , -1)
    pred = np.squeeze(pred ,-1)
    root_path = os.path.join(path , 'train_pred_pic')
    if plot_num > data.shape[1]:
        plot_num = data.shape[1]
    #当前采样
    samples_no = sample_without_put_back(np.arange(data.shape[1]) , plot_num)
    current_dir = os.path.join(root_path, 'epoch({})'.format(epoch))
    if not os.path.isdir(current_dir):
        os.makedirs(current_dir)
    for ssm_no in np.arange(data.shape[0]):
        for sample in samples_no:
            pic_name = os.path.join(current_dir, 'batch_no({})_dataset({})_sample({})'.format(batch, targets_name.split(',')[ssm_no], sample))
            time_range = pd.date_range(time_start[sample], periods=data.shape[2], freq=freq)
            time_range = time_range[-plot_length:]
            s1 = data[ssm_no , sample ,-plot_length:]
            s1[s1 == nan_data] = np.NaN
            s2 = pred[ssm_no , sample ,-plot_length:]
            fig = plt.figure(figsize=(28, 21))

            ax = fig.add_subplot(1, 1, 1)
            ax.set_title('DATASET({}) EPOCHE({} prediction result)'.format(targets_name.split(',')[ssm_no], epoch), fontsize=30)
            ax.plot(time_range, s1, linestyle='-', color='tab:green', marker='D', label='Truth')
            ax.plot(time_range, s2, linestyle='-.', color='tab:blue', marker='o', label='pred_mean')
            ax.xaxis.set_tick_params(labelsize=21)
            ax.yaxis.set_tick_params(labelsize=21)
            ax.legend(prop={'size': 31}, edgecolor='red', facecolor='#e8dfdf')
            plt.savefig(pic_name)
            plt.close(fig)

def plot_train_pred_NoSSMnum(path, targets_name, data, pred, batch, epoch, ssm_no, plot_num, plot_length, time_start, freq, nan_data=0):
    '''
    与上述函数相比，不同的是，data以及pred都没有ssm_num维
    :param path: 存放图片的地址
    :param targets_name 数据集的名
    :param data: 真实数据 #( bs , seq , 1)
    :param pred: 模型训练时预测的 每个时间步的 期望, 维度同 data
    :param batch: 辨明当前的
    :param epoch:当前的epoch
    :param plot_num: 当前情况下需要输出的图片数量
    :param plot_length: 画图序列的长度
    :param time_start : 绘图的初始时间
    :param freq : 当前的时间间隔
    :return:
    '''
    #先把多余的维度去除
    data = np.squeeze(data , -1) #(bs, seq)
    pred = np.squeeze(pred ,-1)
    root_path = os.path.join(path , 'train_pred_pic')
    if plot_num > data.shape[1]:
        plot_num = data.shape[1]
    #当前采样
    samples_no = sample_without_put_back(np.arange(data.shape[0]) , plot_num)
    current_dir = os.path.join(root_path, 'epoch({})'.format(epoch))
    if not os.path.isdir(current_dir):
        os.makedirs(current_dir)
    for sample in samples_no:
        pic_name = os.path.join(current_dir , 'batch_no({})_ssm_no({})_sample({})'.format(batch,ssm_no,sample))
        time_range = pd.date_range(time_start[sample], periods=data.shape[1], freq=freq)
        time_range = time_range[-plot_length:]
        s1 = data[sample, -plot_length:]
        s1[s1 == nan_data] = np.NaN
        s2 = pred[sample , -plot_length:]

        fig = plt.figure(figsize = (28,21))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('DATASET({}) EPOCHE({} prediction result)'.format(targets_name.split(',')[ssm_no], epoch), fontsize=30)
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
        path: str,
        time: str
):
    epoches = list(result.keys())
    loss = list(result.values())
    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epoches, loss, color='tab:blue' , marker='D')

    font = {
        'family': 'Times New Roman',
         'weight': 'normal',
         'size': 30,
    }
    title_font ={
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 40,
    }
    ax.set_xlabel('epoch(s)' ,font )
    ax.set_ylabel('loss(MSE / filter negative log likelihood)' , font)
    ax.set_title('epoch information when training' ,title_font )
    ax.xaxis.set_tick_params(labelsize=21)
    ax.yaxis.set_tick_params(labelsize=21)
    plt.savefig(os.path.join(path , 'train_%s.pdf'%(time)),format="pdf")
    plt.close(fig)

def complete_batch(batch,batch_size):
    # 如果这里不复制，则内存中还是只有一片空间，出现指针混乱！
    completed_batch = batch.copy()
    # 对每一个 key都进行处理
    if len(completed_batch[FieldName.START]) == batch_size:
        return completed_batch , batch_size
    for attr in completed_batch.keys():
        batch_value = completed_batch[attr]
        if isinstance(batch_value , list):
            batch_num = len(batch_value)
            completed_batch[attr] = batch_value + [batch_value[-1]]*(batch_size-batch_num)
        elif isinstance(batch_value, np.ndarray):
            #表示所有非四维的
            if len(batch_value.shape) != TARGET_DIM:
                batch_num = batch_value.shape[0]
                complete_shape = [batch_size-batch_num]+list(batch_value.shape[1:])
                completed_batch[attr] = np.concatenate([batch_value , np.zeros(shape=complete_shape)], axis=0)
            else:
                batch_num = batch_value.shape[1]
                complete_shape = [batch_value.shape[0],batch_size - batch_num] + list(batch_value.shape[2:])
                completed_batch[attr] = np.concatenate([batch_value, np.zeros(shape=complete_shape)], axis=1)
    return completed_batch ,batch_num


def del_previous_model_params(path):
    '''
    :param path:  该组参数组的主目录
    :return:  删除之前epoch的参数文件
    '''
    if not os.path.isdir(path):
        return
    for file in os.listdir(path):
        if file.endswith(".data-00000-of-00001") \
                or file.endswith(".index") \
                or file.endswith(".meta"):
            os.remove(os.path.join(path,file))

def get_model_params_name(path):
    '''
    :param path: 改组参数组的目录 ..../model_params
    :return: 参数文件名
    '''
    if not os.path.isdir(path):
        return
    for file in os.listdir(path):
        if file.endswith(".data-00000-of-00001") \
                or file.endswith(".index") \
                or file.endswith(".meta"):
            params_name = os.path.splitext(file)[0]
            return params_name

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
    ax[1].legend(["prophet_compared series", "end of train series"], loc="upper left")

    plt.savefig('pic/original_entry_{}.png'.format(no))

# 给定序列的均值和方差，进行采样：
def samples_with_mean_cov(mean: np.ndarray , cov:np.ndarray , num_samples: int):
    '''
    :param mean: (ssm_num, bs, pred, dim_z)
    :param cov: (ssm_num, bs, pred, dim_z ， dim_z)
    :return: samples (ssm_num ,bs,num_samples, pred, dim_z)
    '''
    result = np.zeros(shape=mean.shape)
    result =  np.tile(np.expand_dims(result,0),[100]+[1]*len(result.shape))#(samples, ssm ,bs, pred, dim_z)

    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            for k in range(mean.shape[2]):
                samples = np.random.multivariate_normal(mean[i,j,k] , cov[i,j,k] , size=num_samples)
                result[: , i, j , k] = samples
                # print(samples.shape)
    result = np.transpose(result ,[1,2,0,3,4])
    return result
    pass

'''
根据要 reload 的path 修改 config的值
'''
def get_reload_hyper(path, config):
       
    abbre = {
        'freq' : 'freq',
        'env' : 'environment',
        'lags' : 'maxlags',
        'past' : 'past_length',
        'pred' : 'pred_length',
        'u' : 'dim_u',
        'l' : 'dim_l',
        'K' : 'K',
        'T' : ['time_exact_layers' , 'time_exact_cells'],
        'E' : ['env_exact_layers' , 'env_exact_cells'],
        'α' : 'alpha_units',
        'epoch' : 'epochs',
        'bs' : 'batch_size',
        'bn' : 'num_batches_per_epoch',
        'lr' : 'learning_rate' ,
        'initKF' : 'init_kf_matrices',
        'dropout':'dropout_rate'
    }

    not_convert_to_int = ['freq' ,'env' , 'T' , 'E']
    convert_to_float = ['lr' , 'initKF' , 'dropout']
    hyper_parameters = path.split('_');
    for parameter in hyper_parameters:
        index_of_left_embrace = parameter.index('(')
        index_of_right_embrace = parameter.index(')')
        name = parameter[:index_of_left_embrace]
        value = parameter[index_of_left_embrace+1 : index_of_right_embrace]
        if name in convert_to_float :
            value = float(value)
        elif name not in not_convert_to_int:
            value = int(value)
        else:
            #维持原样就好了
            pass
        
        #单参数
        if isinstance(abbre[name] , str):
            config.__setattr__(abbre[name] , value)
        #多参数，形容LSTM
        else:
            value_split = value.split('-')
            for i in range(len(abbre[name])):
                config.__setattr__(abbre[name][i] , int(value_split[i]))
    
    return config
        