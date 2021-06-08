#!/bin/bash
#source activate gluonts
#frozen
source activate gluonts
# 比特币数据
logs_dir='logs/btc_eth(deep_state)'
target='btc,eth'
start='2018-08-02'
length='503'
freq='1D'

#重金属数据
#logs_dir='logs/metal_overlap(deep_state)'
#target="LMEAluminium,LMECopper,LMELead,LMENickel,LMETin,LMEZinc"
#start='2017-01-02'
#length='782'
#freq='1B'



slice='overlap'
batch_size='32'
batch_num='22'
epoch='100'

past='90'
pred='5'

for ((i=1;i<=6;i=i+1))#! repettion
do
    python gluonts/lzl_shared_ssm/models/lstm_compared/run_lstm.py --target=$target --start=$start --timestep=$length --freq=$freq \
    --slice=$slice --batch_size=$batch_size --num_batches_per_epoch=$batch_num --epochs=$epoch --past_length=$past --pred_length=$pred \
    --logs_dir=$logs_dir
    # python gluonts/lzl_shared_ssm/models/deepstate_compared/run_deep_state.py --target=$target --start=$start --timestep=$length --freq=$freq \
    # --slice=$slice --batch_size=$batch_size --num_batches_per_epoch=$batch_num --epochs=$epoch --past_length=$past --pred_length=$pred \ 
    # --logs_dir=$logs_dir
    python gluonts/lzl_shared_ssm/models/prophet_compared/test_prophet.py -t=$target -st=$start -f=$freq -T=$length -pr=$pred -pa=$past -s=$slice
    # python gluonts/lzl_shared_ssm/models/ARIMA_compared/run_auto_arima.py -t=$target -st=$start -f=$freq -T=$length -pr=$pred -pa=$past -s=$slice
    # python gluonts/lzl_shared_ssm/models/deepar_compared/run_deepar.py -t=$target -st=$start -f=$freq -T=$length -pr=$pred -pa=$past -s=$slice -bs=$batch_size -bn=$batch_num -ep=$epoch
    
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
