#!/bin/bash
source activate gluonts
past='60'
pred='1'

# 选择不同数据集对应的log
indice_lstm_logs_dir='logs/ukx_vix_spx_shsz_nky_train_overlap(lstm_202107)'
crypto_lstm_logs_dir='logs/btc_eth_overlap(lstm)'
air_lstm_logs_dir='logs/air_overlap(lstm)'

# air quality
AIR_target="PM25,PM10,NO2"
AIR_start='2013-4-15_17:00:00'
AIR_length='1168'
AIR_freq='1H'

# economic indice
E_target="UKX,VIX,SPX,SHSZ300,NKY"
E_start='2008-01-04'
E_length='2606'
E_freq='1B'

# btc and eth
target='btc,eth'
start='2018-08-02'
length='503'
freq='1D'

slice='overlap'
batch_size='32'
epoch='100'


# btc and eth
batch_num='14'
for ((i=0;i<3;i=i+1))#! repettion
do
   python gluonts/lzl_shared_ssm/models/lstm_compared/run_lstm.py --target=$target --start=$start --timestep=$length --freq=$freq \
   --slice=$slice --batch_size=$batch_size --num_batches_per_epoch=$batch_num --epochs=$epoch --past_length=$past --pred_length=$pred \
   --logs_dir=$crypto_lstm_logs_dir &
done
# E_batch_num='80'
# for ((i=0;i<6;i=i+1))#! repettion
# do
    
#     # cpu_now=$(($cpu+$i*2+1))
#     python gluonts/lzl_shared_ssm/models/lstm_compared/run_lstm.py --target=$E_target --start=$E_start --timestep=$E_length --freq=$E_freq \
#     --slice=$slice --batch_size=$batch_size --num_batches_per_epoch=$E_batch_num --epochs=$epoch --past_length=$past --pred_length=$pred \
#     --logs_dir=$indice_lstm_logs_dir 
# done

# AIR_batch_num='35'
# for ((i=0;i<1;i=i+1))#! repettion
# do
    
#     python gluonts/lzl_shared_ssm/models/lstm_compared/run_lstm.py --target=$AIR_target --start=$AIR_start --timestep=$AIR_length --freq=$AIR_freq \
#     --slice=$slice --batch_size=$batch_size --num_batches_per_epoch=$AIR_batch_num --epochs=$epoch --past_length=$past --pred_length=$pred \
#     --logs_dir=$air_lstm_logs_dir 
# done