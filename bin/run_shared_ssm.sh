#!/bin/bash
source activate gluonts
# btc eth 数据集
(90,5,13) (60,3,14) (30,1,15)
logs_dir="logs/btc_eth(shared_ssm)"
reload_model=''
reload_time=''
target="btc,eth"
environment='gold'
use_env='True'
start='2018-08-02'
timestep='503'
freq='1D'
slice='overlap'

# indices 数据集
#(30,1,81) (60,3,80) (90,5,79) ===> 2606
#(30,1,89) (60,3,88) (90,5,87) ===> 2867
# use_env='True'
# logs_dir="logs/ukx_vix_spx_shsz_nky_train_overlap(shared_ssm)"
# reload_model=''
# reload_time=''
# target="UKX,VIX,SPX,SHSZ300,NKY"
# environment='SX5E,DXY'
# start='2008-01-04'
# timestep='2606'
# freq='1B'
# slice='overlap'


past_length='30'
prediction_length='3'
batch_num='81'
batch_size='32'
epochs='100'
num_samples='100'
#之前的版本u为10 l是5
dim_u='5'
dim_l='4'
initializer="xavier"



#USE FOR TRAINIG
for K in 2
do
  for maxlags in 3 4 5 6 7
  do
    for ((i=1;i<=3;i=i+1))#! repettion
    do
        for drop_prob in 0.5
        do
            for lr in 0.001
            do
            python gluonts/lzl_shared_ssm/run_shared_ssm.py --reload_model=$reload_model --reload_time=$reload_time --logs_dir=$logs_dir \
            --dropout_rate=$drop_prob --learning_rate=$lr --target=$target --environment=$environment --maxlags=$maxlags \
            --start=$start --timestep=$timestep --past_length=$past_length --pred_length=$prediction_length \
            --slice=$slice --batch_size=$batch_size --num_batches_per_epoch=$batch_num --epochs=$epochs \
            --use_env=$use_env --num_samples=$num_samples --freq=$freq --dim_l=$dim_l --dim_u=$dim_u --K=$K \
            --initializer=$initializer 
          
            done
        done
    done
  done
done

# USED FOR RELOAD MODEL
# for reload_time in '' 1 2 3 4 5
# do
#     python gluonts/lzl_shared_ssm/run_shared_ssm.py --reload_model=$reload_model --reload_time=$reload_time --logs_dir=$logs_dir \
#     --target=$target --environment=$environment --start=$start --timestep=$timestep --past_length=$past_length \
# --pred_length=$prediction_length --slice=$slice --batch_size=$batch_size --num_batches_per_epoch=$batch_num --epochs=$epochs \
#     --use_env=$use_env --num_samples=$num_samples --freq=$freq --dim_l=$dim_l --dim_u=$dim_u --K=$K
# done

#! [ command ] 在判断语句前后都要空格
# 如果需要筛选某些超参，可以在这里进行判断选择
# if [ $K -eq 4 ] && [ $maxlags -eq 6 ]; then
#     echo ignore
#     continue;
# fi