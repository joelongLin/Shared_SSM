#!/bin/bash
#cd /home/cjw/MTS_DA
#source activate tensorflow1.9_env
#frozen
dataset="btc_eth"
logs_dir="logs/"$dataset
past_length="7"
prediction_length='1'
add_trend='True'
reload_model='logs/btc_eth/Dec_27_18:00:05_2019'
#1
#model="Model3_MMD"
#model_save_dir_prefix="5to10"
#src_train_path="5/train.csv"
#src_test_path="5/test.csv"
#tgt_train_path="Boiler_#03_170721.csv"
#tgt_test_path="Boiler_#03_170721.csv"
#2
#cuda=$1
#lr=$2
#drop_prob=$3
for ((i=1;i<=1;i=i+1))#! repettion
do
    for drop_prob in 0.5
    do
        for lr in 0.0005
        do
        python gluonts/lzl_deepstate/run_dssm.py --reload_model=$reload_model --logs_dir=$logs_dir --dropout_rate=$drop_prob --learning_rate=$lr --dataset=$dataset --past_length=$past_length --prediction_length=$prediction_length --add_trend=$add_trend
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
