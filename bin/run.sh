#!/bin/bash
#source activate gluonts
#frozen

for ((i=1;i<=6;i=i+1))#! repettion
do
    python gluonts/lzl_shared_ssm/models/prophet_compared/test_prophet.py
    python gluonts/lzl_shared_ssm/models/lstm_compared/run_lstm.py
    python gluonts/lzl_shared_ssm/models/deepstate_compared/run_deep_state.py

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
