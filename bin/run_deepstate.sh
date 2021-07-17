#!/bin/bash
#source activate gluonts
#frozen

source activate tf_mx
past='60'
pred='3'
slice='overlap'
batch_size='32'
epoch='1'

# btc and eth
#(90,5,13) (60,3,14) (30,1,15)
target='btc,eth'
start='2018-08-02'
length='503'
freq='1D'
env='gold'
batch_num='15'

#economic indice
# (30,1,81) (60,3,80) (90,5,79) batch_size = 32  --> 2606
E_target="UKX,VIX,SPX,SHSZ300,NKY"
E_start='2008-01-04'
E_length='2606'
E_freq='1B'
E_env='SX5E,DXY'
E_batch_num='81'

# air quality
AIR_target="PM25,PM10,NO2"
AIR_start='2013-4-15_17:00:00'
AIR_length='1172'
AIR_freq='1H'
AIR_env='temperature,pressure,humidity,wind'
AIR_batch_num='35'

# without ENV
python gluonts/lzl_shared_ssm/models/deepstate_compared/run_deepstate_gluonts.py --target=$target --start=$start --timestep=$length --freq=$freq \
        --slice=$slice --batch_size=$batch_size --batch_num=$batch_num --epochs=$epoch --past=$past --pred=$pred

# with ENV
for lags in 3 4 5 6 7
do
    for ((i=1;i<=1;i=i+1))#! repettion
    do
        python gluonts/lzl_shared_ssm/models/deepstate_compared/run_deepstate_gluonts_env.py --target=$target --start=$start --timestep=$length --freq=$freq \
        --slice=$slice --batch_size=$batch_size --batch_num=$batch_num --epochs=$epoch --past=$past --pred=$pred --env=$env --lags=$lags
    done
done

# you can also use the re-implemented deepstate in tensorflow1.14 (only for {btc,eth} and {economic indice})
python gluonts/lzl_shared_ssm/models/deepstate_compared/run_deepstate.py --reload_model=$reload_model --reload_time=$reload_time --logs_dir=$logs_dir \
        --dropout_rate=$drop_prob --learning_rate=$lr --target=$target  --start=$start --timestep=$timestep --past_length=$past_length --pred_length=$prediction_length \
        --slice=$slice --batch_size=$batch_size --num_batches_per_epoch=$batch_num --epochs=$epochs --num_eval_samples=$num_samples --freq=$freq

