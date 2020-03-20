#!/bin/bash
#cd /home/lzl/pycharm/gluon
#source activate gluonts
#frozen
gpu='3'
reload_model='freq(1D)_lags(0)_past(90)_pred(5)_u(10)_l(4)_K(3)_T(2-40)_E(2-50)_α(50)_epoch(50)_bs(32)_bn(13)_lr(0.001)_initKF(0.05)_initCov(10.0)_dropout(0.5)'
target="btc,eth"
environment='gold'
logs_dir="logs/btc_eth(shared_ssm)"
start='2018-08-02'
maxlags='0'
past_length='90'
prediction_length='5'
batch_size='32'
batch_num='13'
epochs='50'
for ((i=1;i<=1;i=i+1))#! repettion
do
    for drop_prob in 0.5
    do
        for lr in 0.001
        do
        python gluonts/lzl_shared_ssm/run_main.py --gpu=$gpu --reload_model=$reload_model --logs_dir=$logs_dir \
        --dropout_rate=$drop_prob --learning_rate=$lr --target=$target --environment=$environment --maxlags=$maxlags \
        --start=$start --past_length=$past_length --pred_length=$prediction_length --batch_size=$batch_size \
        --num_batches_per_epoch=$batch_num --epochs=$epochs
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
