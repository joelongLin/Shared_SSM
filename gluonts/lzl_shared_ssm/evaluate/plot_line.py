# -*- coding: UTF-8 -*-
# author : joelonglin
import numpy as np
import pickle
import os
from scipy import stats
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import logging
import sys
sys.path.insert(0,os.getcwd())

if '/evaluate' in os.getcwd():
    os.chdir('../')
if '/lzl_shared_ssm' not in os.getcwd():
    os.chdir('gluonts/lzl_shared_ssm')

# 实验结果选择
# target = 'btc,eth'
target = 'UKX,SPX,SHSZ300,NKY,VIX'
slice_style = 'overlap'
# length=503
length = 2606
start = '2008-06-30'
freq = '1B'
past = 90
seq = 5
maxlags = 6
dim_l = 4
dim_u = 5
dim_z = 1


root_path = 'data_process/raw_data/'
pic_root_path = os.path.join(root_path , 'data_plot')
if not os.path.exists(pic_root_path):
    os.mkdir(pic_root_path)

data_root_path = 'data_process/raw_data/SIGIR/Train/Train_data/'



if __name__ == '__main__':
    #Indices_DXY Curncy_train.csv
    #针对不同的数据集进行绘图
    fig_trgt, ax_trgt = plt.subplots(
                # figsize 表达方式是 长 × 宽 这里还是与pandas中的数据有点不同的
                figsize=(30 , 4), ncols=len(target.split(",")), nrows=1, constrained_layout=True, sharex=True , sharey= False
    )
    print(type(ax_trgt[0]))
    item_no = 0;
    for item in target.split(','):
        item_no += 1
        curr_file = data_root_path  + 'Indices_{} Index_train.csv'.format(item)
        item_series = pd.read_csv(curr_file, usecols=(1,2), header=0, index_col=0);
        item_series.index = pd.DatetimeIndex(item_series.index)
        item_series = item_series.loc[start:, item]
        # item_series = item_series / item_series.mean()
        print("成功了");
        print(item_series)
        # print(item_pd.head())
        # print(item_pd.index)
        ax_trgt[item_no-1].plot(item_series)
        ax_trgt[item_no-1].set_title(item, fontsize=35)
    
    plt.savefig(os.path.join(pic_root_path ,'data.pdf'))
        
        
        
        
    exit();
       
                    

                   
    
#//curr_file = data_root_path  + '{}_start({})_freq({})_{}_DsSeries_{}_train_{}_pred_{}.pkl'.format(item, start, freq, slice_style, length+1-past-seq, past, seq)
#//with open(curr_file ,'rb') as fp:
#//    curr_data = pickle.load(fp);
            
                
    


