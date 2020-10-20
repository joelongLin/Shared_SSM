#!/bin/bash
ip_result=`/sbin/ifconfig -a|grep inet|grep -v inet6|grep -v 127.0.0.1|grep -E '10.21.25.*'|awk '{print $2}'|tr -d "addr:"`
echo 'current ip is : '$ip_result
source activate gluonts

gpu='3'

# btc eth 数据集
# logs_dir="logs/btc_eth(shared_ssm)"
# reload_model=''
# reload_time=''
# target="btc,eth"
# environment='gold'
# use_env='False'
# start='2018-08-02'
# timestep='503'
# freq='1D'
# slice='overlap'

# metal 数据集
logs_dir="logs/metal_overlap(shared_ssm)"
reload_model='freq(1B)_env(comexCopper,comexGold,comexPalladium,comexPlatinum,comexSilver)_lags(5)_past(90)_pred(5)_u(5)_l(4)_K(2)_T(2-40)_E(2-50)_α(50)_epoch(100)_bs(32)_bn(22)_lr(0.001)_initKF(0.05)_dropout(0.5)'
reload_time=''
target="LMEAluminium,LMECopper,LMELead,LMENickel,LMETin,LMEZinc"
environment='comexCopper,comexGold,comexPalladium,comexPlatinum,comexSilver'
use_env='True'
start='2017-01-02'
timestep='782'
freq='1B'
slice='overlap'

dim_u='10'
dim_l='5'
past_length='90'
prediction_length='5'
batch_size='32'
batch_num='22'
epochs='100'
num_samples='100'
for maxlags in 5 #5 4 3 2 1 0
do
  for ((i=1;i<=6;i=i+1))#! repettion
  do
      for drop_prob in 0.5
      do
          for lr in 0.001
          do
          python gluonts/lzl_shared_ssm/run_shared_ssm.py --gpu=$gpu --reload_model=$reload_model --reload_time=$reload_time --logs_dir=$logs_dir \
          --dropout_rate=$drop_prob --learning_rate=$lr --target=$target --environment=$environment --maxlags=$maxlags \
          --start=$start --timestep=$timestep --past_length=$past_length --pred_length=$prediction_length \
          --slice=$slice --batch_size=$batch_size --num_batches_per_epoch=$batch_num --epochs=$epochs \
          --use_env=$use_env --num_samples=$num_samples --freq=$freq --dim_l=$dim_l --dim_u=$dim_u
          done
      done
  done
done
#python train.py -cuda_device=$1 -lr=0.0003 -drop_prob=0.3 -repetition_idx=3 -model=$model -model_save_dir_prefix=$model_save_dir_prefix -src_train_path=$src_train_path -src_test_path=$src_test_path -tgt_train_path=$tgt_train_path -tgt_test_path=$tgt_test_path
#python train.py -cuda_device=$1 -lr=0.0002 -drop_prob=0.3 -repetition_idx=3 -model=$model -model_save_dir_prefix=$model_save_dir_prefix -src_train_path=$src_train_path -src_test_path=$src_test_path -tgt_train_path=$tgt_train_path -tgt_test_path=$tgt_test_path
#for lr in 0.0005 0.0004 0.0003 0.0002
#do
    #python train.py -cuda_device=$1 -lr=$lr -drop_prob=0.2 -repetition_idx=3 -model=$model -model_save_dir_prefix=$model_save_dir_prefix -src_train_path=$src_train_path -src_test_path=$src_test_path -tgt_train_path=$tgt_train_path -tgt_test_path=$tgt_test_path
#done

#for lr in 0.0005 0.0004 0.0003 0.0002
#do
#    python train.py -cuda_device=$1 -lr=$lr -drop_prob=0.3 -repetition_idx=3 -model=$model -model_save_dir_prefix=$model_save_dir_prefix -src_train_path=$src_train_path -src_test_path=$src_test_path -tgt_train_path=$tgt_train_path -tgt_test_path=$tgt_test_path
#done
#python train.py -cuda_device=$1 -lr=0.0002 -drop_prob=0.3 -repetition_idx=2 -model=$model -model_save_dir_prefix=$model_save_dir_prefix -src_train_path=$src_train_path -src_test_path=$src_test_path -tgt_train_path=$tgt_train_path -tgt_test_path=$tgt_test_path
#python train.py -cuda_device=$1 -lr=0.0002 -drop_prob=0.2 -repetition_idx=2 -model=$model -model_save_dir_prefix=$model_save_dir_prefix -src_train_path=$src_train_path -src_test_path=$src_test_path -tgt_train_path=$tgt_train_path -tgt_test_path=$tgt_test_path