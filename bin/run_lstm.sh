#!/bin/bash
source activate gluonts
# cpu='0'
past='60'
pred='1'

# (30,1,81) (60,3,80) (90,5,79) batch_size = 32
indice_lstm_logs_dir='logs/ukx_vix_spx_shsz_nky_train_overlap(lstm)'
crypto_lstm_logs_dir='logs/btc_eth_overlap(lstm)'

E_target="UKX,VIX,SPX,SHSZ300,NKY"
E_start='2008-01-04'
E_length='2606'
E_freq='1B'

target='btc,eth'
start='2018-08-02'
length='503'
freq='1D'

slice='overlap'
batch_size='32'
epoch='100'


#batch_num='14'

#for ((i=0;i<3;i=i+1))#! repettion
#do
#    python gluonts/lzl_shared_ssm/models/lstm_compared/run_lstm.py --target=$target --start=$start --timestep=$length --freq=$freq \
#    --slice=$slice --batch_size=$batch_size --num_batches_per_epoch=$batch_num --epochs=$epoch --past_length=$past --pred_length=$pred \
#    --logs_dir=$crypto_lstm_logs_dir &
    
#done

#wait

E_batch_num='80'
for ((i=0;i<6;i=i+1))#! repettion
do
    
    # cpu_now=$(($cpu+$i*2+1))
    python gluonts/lzl_shared_ssm/models/lstm_compared/run_lstm.py --target=$E_target --start=$E_start --timestep=$E_length --freq=$E_freq \
    --slice=$slice --batch_size=$batch_size --num_batches_per_epoch=$E_batch_num --epochs=$epoch --past_length=$past --pred_length=$pred \
    --logs_dir=$indice_lstm_logs_dir 
done

