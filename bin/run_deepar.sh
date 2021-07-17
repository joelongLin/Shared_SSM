#!/bin/bash
#source activate gluonts
#frozen
source activate tf_mx
past='60'
pred='3'
slice='overlap'
batch_size='32'
epoch='100'

# btc and eth
#(90,5,13) (60,3,14) (30,1,15)
target='btc,eth'
start='2018-08-02'
length='503'
freq='1D'
env='gold'
batch_num='14'


#economic indice
# (30,1,81) (60,3,80) (90,5,79) batch_size = 32  --> 2606
E_target="UKX,VIX,SPX,SHSZ300,NKY"
E_start='2008-01-04'
E_length='2867'
E_freq='1B'
E_env='SX5E,DXY'
E_batch_num='88'

# air quality
AIR_target="PM25,PM10,NO2"
AIR_start='2013-4-15_17:00:00'
AIR_length='1172'
AIR_freq='1H'
AIR_env='temperature,pressure,humidity,wind'
AIR_batch_num='35'



# without ENV
python gluonts/lzl_shared_ssm/models/deepar_compared/run_deepar.py -t=$target -st=$start -f=$freq -T=$length -pr=$pred -pa=$past -s=$slice -bs=$batch_size -bn=$batch_num -ep=$epoch  

# with ENV
for lags in 1 2 3 4 5
do
    for ((i=1;i<=1;i=i+1))#! repettion
    do 
        python gluonts/lzl_shared_ssm/models/deepar_compared/run_deepar_env.py -e=$env -l=$lags -t=$target -st=$start -f=$freq -T=$length -pr=$pred -pa=$past -s=$slice -bs=$batch_size -bn=$batch_num -ep=$epoch  
    done
done