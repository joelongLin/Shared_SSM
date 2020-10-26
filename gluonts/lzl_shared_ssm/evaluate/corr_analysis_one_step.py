# -*- coding: UTF-8 -*-
# author : joelonglin
#* 本文档主要是根据蔡老师的要求，只单单比较第一个时间步
import numpy as np
import pickle
import os
from scipy import stats
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging

if '/evaluate' in os.getcwd():
    os.chdir('../')
if '/lzl_shared_ssm' not in os.getcwd():
    os.chdir('gluonts/lzl_shared_ssm')

# 实验结果选择
# target = 'btc,eth'
target = 'UKX,VIX,SPX,SHSZ300,NKY'
slice_style = 'overlap'
# length=503
length = 2867
past = 90
seq = 5
maxlags = 5
dim_l = 4
dim_u = 5
dim_z = 1

compared_var_dim = {
    'ground_truth' : dim_z,
    'l_mean' : dim_l,
    'z_mean_scaled' : dim_z,
    'control' : dim_u
}
compared_var_math = {
    'ground_truth' : r"$z$",
    'l_mean' : r"$l$",
    'z_mean_scaled' : r"$\hat{z}$",
    'control' : r"$u$"
}
#? 现在看样子与 compared_var_dim 一样
compared_plot_length = {
    'ground_truth' : dim_z,
    'l_mean' : dim_l,
    'z_mean_scaled' : dim_z,
    'control' : dim_u
}
# 从 compared_var_dim 中选择
compared_1 = 'l_mean'
compared_2 = 'control'

# 是否输出热点图
draw_heat_map = True




root_path = 'evaluate/analysis/{}_length({})_slice({})_past({})_pred({})'.format(target.replace(',' ,'_') , length , slice_style ,past , seq)
pic_root_path = os.path.join(root_path , 'corr_analysis_pic')
if not os.path.exists(pic_root_path):
    os.mkdir(pic_root_path)

prefix = 'seq({})_dim_l({})_dim_u({})_lag({})'.format(
            seq ,dim_l , dim_u , maxlags
        )


def get_ticks(variable : str , i : int) -> str:
    ticks_prefix = compared_var_math[variable]
    ticks_prefix_list = list(ticks_prefix)
    ticks_prefix_list.insert(ticks_prefix.find("$" ,1 ) , "_" + str(i+1))
    ticks = "".join(ticks_prefix_list)
    return ticks



if __name__ == '__main__':
    min_pvalue = float("inf")
    min_pvalue_file = ""
    for path in os.listdir(root_path):
        if path != "seq(5)_dim_l(4)_dim_u(5)_lag(5).pkl":
            continue
        if prefix in path:
            file_marker = path.strip(prefix).split('.')[0]
            # 生成的代码里面没有(0) 所以只能自己加了
            if file_marker == '':
                file_marker = "time(0)"
            print('\n\n==================== 当前分析的文件是 {} ============================'.format(path))
            with open(os.path.join(root_path , path) , 'rb') as fp:
                prediction_analysis =  pickle.load(fp) # * (ssm , samples , seq , dim)
                #* corr_heatmap会放 num_items 个热点图
                corr_heatmap = []; pvalue_heatmap = []
                item_nums = prediction_analysis['l_mean'].shape[0]
                # 面对不同的 target 
                for i_target in range(item_nums):
                    
                    corr_input = np.concatenate(
                        (prediction_analysis[compared_1][i_target ,  : , 0 , :],
                        prediction_analysis[compared_2][i_target ,  : , 0 , :] ),
                        axis=-1  
                    )#(samples , dim_l(dim_z) + dim_u)
                    dim_com_1 = compared_var_dim[compared_1]
                    dim_com_2 = compared_var_dim[compared_2]
                    corr_output = np.zeros((dim_com_1 , dim_com_2))
                    pvalue_output = np.zeros((dim_com_1 , dim_com_2))
                    for i , j in itertools.product(range(dim_com_1) , range(dim_com_2)):
                        corr_ij , p_ij = stats.spearmanr(corr_input[:,i] , corr_input[:, dim_com_1 + j])
                        corr_output[i ,j] = corr_ij
                        pvalue_output[i,j] =  p_ij
                    
                    print('\n{}序列的第1个时间步相关性分析如下(行为{} , 列为{})'.format(target.split(',')[i_target] , compared_1 , compared_2))
                    print(corr_output)
                    print(pvalue_output)
                    print("相关系数的绝对值和为：",np.sum(abs(corr_output)))
                    
                    corr_heatmap.append(corr_output)
                    pvalue_heatmap.append(pvalue_output)
                
                curr_pvalue = 0;
                for i in range(item_nums):
                    curr_pvalue += np.sum(abs(pvalue_heatmap[i]))
                print("当前的pvalue为：" , curr_pvalue)
                if curr_pvalue <= min_pvalue:
                    print('更新pvalue...')
                    min_pvalue = curr_pvalue
                    min_pvalue_file = path;
                #! 不再独个结果地呈现相关性，而是呈现每个文件 5个序列的长度
                if draw_heat_map:
                    #? 给图片的长度添加补全操作，因为横坐标上的内容比较多一些
                    offset = 5
                    x_ticks = [ get_ticks(compared_1 , i)
                                for i in range(compared_var_dim[compared_1])]
                            
                    y_ticks = [ get_ticks(compared_2 , i)
                                for i in range(compared_var_dim[compared_2])]
                    
                    corr_vmax = 0.4; corr_vmin = -0.4
                    pvalue_vmax = 1.0 ; pvalue_vmin = 0.0;

                    #* 先绘出相关系数
                    fig_corr, ax_corr = plt.subplots(
                        # figsize 表达方式是 长 × 宽 这里还是与pandas中的数据有点不同的
                        figsize=(compared_var_dim[compared_2]*item_nums+offset , compared_plot_length[compared_1]),
                        ncols=item_nums, nrows=1, constrained_layout=True, sharex=True , sharey= True
                    )
                    
                    # cmap = sns.cm.rocket_r
                    cmp = "YlGnBu"

                    #? 第二行的有 x_ticks 最左边的有 y_ticks 最右边有colorbar
                    for i in range(item_nums):
                        print('正在绘制' + target.split(",")[i] + '第一个时间步的结果...')    
                        corr_pd_data=pd.DataFrame(corr_heatmap[i] , index=x_ticks,columns=y_ticks)
                        sns.heatmap(corr_pd_data ,annot=False, annot_kws={"size":10} , vmax=corr_vmax, vmin=corr_vmin 
                                            , fmt=".2f" ,cmap=cmp, cbar=(True if i==item_nums-1 else False), ax = ax_corr[i]
                        )
                        #单独设置colorbar的大小
                        if i == item_nums-1:
                            ax_corr[i].collections[0].colorbar.ax.tick_params(labelsize=25)
                        ax_corr[i].tick_params(labelsize=20) 
                        #在头部设置标题
                        ax_corr[i].set_title(target.split(",")[i],fontsize=35)
                            
                    plt.savefig(os.path.join(pic_root_path ,
                        '1st_step_{}_lags({})_{}_{}_vs_{}.pdf'.format(
                            target.replace("," ,"_and_") ,maxlags ,file_marker,  compared_1 , compared_2
                            ))
                        )
                    

                    #* 用于绘制pvalue的图
                    fig_pvalue, ax_pvalue = plt.subplots(
                         # figsize 表达方式是 长 × 宽 这里还是与pandas中的数据有点不同的
                        figsize=(compared_var_dim[compared_2]*item_nums+offset , compared_plot_length[compared_1]),
                        ncols=item_nums, nrows=1, constrained_layout=True, sharex=True , sharey= True
                    )

                    for i in range(item_nums):
                        print('正在绘制' + target.split(",")[i] + '第一个时间步的pvalue热点图...')
                        
                            
                        pvalue_pd_data=pd.DataFrame(pvalue_heatmap[i] , index=x_ticks,columns=y_ticks)
                        sns.heatmap(pvalue_pd_data ,annot=False, annot_kws={"size":10} , vmax=pvalue_vmax, vmin=pvalue_vmin 
                                            , fmt=".2f" ,cmap=cmp, cbar=(True if i==seq-1 else False), ax = ax_pvalue[i]
                        )
                         #单独设置colorbar的大小
                        if i == item_nums-1:
                            ax_pvalue[i].collections[0].colorbar.ax.tick_params(labelsize=25)
                        ax_pvalue[i].tick_params(labelsize=20) 
                        
                        #在头部设置时间步
                        
                        ax_pvalue[i].set_title(target.split(",")[i],fontsize=35)
                    
                    plt.savefig(os.path.join(pic_root_path ,
                        '1st_step_{}_lags({})_{}_{}_vs_{}_pvalue.pdf'.format(
                            target.replace("," ,"_and_") ,maxlags ,file_marker,  compared_1 , compared_2
                            ))
                        )

            print('========================================================== end==================================================\n')    
                    #// ax0 = sns.heatmap(corr_pd_data, annot=True, annot_kws={"size":20} ,vmax=0.5,vmin=0.0, ax=ax0, fmt='.2f', cmap=cmap)
                    #// ax0.set_title('{} series {} vs {} , step {} spearman correlation [{}]'.format(target.split(',')[i_target], compared_1 , compared_2 , i_step+1, file_marker))
                    #// ax0.tick_params(labelsize=20)
                    #// ax0.collections[0].colorbar.ax.tick_params(labelsize=20)
                    #// 绘出pvalue值
                    #// ax1 = sns.heatmap(pvalue_pd_data, annot=True, annot_kws={"size":20} ,  ax=ax1, fmt='.2f', cmap=cmap)
                    #// ax1.set_title('{} series {} vs {} , step {} p value [{}]'.format(target.split(',')[i_target] , compared_1 , compared_2 , i_step+1, file_marker))
                    #// ax1.tick_params(labelsize=20)                          
                    #// ax1.collections[0].colorbar.ax.tick_params(labelsize=20)
                                                

                    
                    #// plt.savefig(os.path.join(pic_root_path ,
                    #//     '{}-series_{}_step_{}_{} vs {}.png'.format(
                    #//         file_marker,target.split(',')[i_target] , i_step+1 , compared_1 , compared_2
                    #//         ))
                    #//     )
                    #// plt.close()
                    #// exit()
    print("当前最小pvalue的文件为 " , min_pvalue_file ,"其pvalue的值为" , min_pvalue)

            
                
    


