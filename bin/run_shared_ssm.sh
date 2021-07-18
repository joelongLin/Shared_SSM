#!/bin/bash

# btc eth 
#(90,5,13) (60,3,14) (30,1,15)
# Scale of the GOLD are relatively small,so the noise should be 1 to 2.5
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
past_length='60'
prediction_length='1'
batch_num='14'

# economic indices
#(30,1,81) (60,3,80) (90,5,79) ===> 2606
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
# past_length='60'
# prediction_length='1'
# batch_num='80'

# air quality 数据集
# (60, {1,3,5}, 35)
# use_env='True'
# logs_dir="logs/air_quality(shared_ssm)"
# reload_model=''
# reload_time=''
# target="PM25,PM10,NO2"
# environment='temperature,pressure,humidity,wind'
# start='2013-4-15_21:00:00' # 日期这里面的空格换成下划线
# timestep='1168'
# freq='1H'
# slice='overlap'
# past_length='60'
# prediction_length='3'
# batch_num='35'

batch_size='32'
epochs='100'
num_samples='100'
dim_u='5'
dim_l='4'
drop_prob='0.5'
lr='0.001'


#USE FOR TRAINIG
for K in 2 3 4 5
do
  for maxlags in 3 4 5 6 7
  do
    for ((i=1;i<=6;i=i+1))#! repettion
    do
      for env_noise in 0
        do
          python gluonts/lzl_shared_ssm/run_shared_ssm.py --reload_model=$reload_model --reload_time=$reload_time --logs_dir=$logs_dir \
          --dropout_rate=$drop_prob --learning_rate=$lr --target=$target --environment=$environment --maxlags=$maxlags \
          --start=$start --timestep=$timestep --past_length=$past_length --pred_length=$prediction_length \
          --slice=$slice --batch_size=$batch_size --num_batches_per_epoch=$batch_num --epochs=$epochs \
          --use_env=$use_env --num_samples=$num_samples --freq=$freq --dim_l=$dim_l --dim_u=$dim_u --K=$K \
          --env_noise=$env_noise --env_noise_mean=0.0
        done
    done
  done
done
