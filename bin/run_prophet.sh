#!/bin/bash
source activate gluonts
past='60'
pred='3'
slice='overlap'

# btc and eth
target='btc,eth'
start='2018-08-02'
length='503'
freq='1D'
env='gold'


# economic indice
E_target='UKX,VIX,SPX,SHSZ300,NKY'
E_start='2008-01-04'
E_length='2606'
E_freq='1B'
E_env='SX5E,DXY'

# air quality
AIR_target='PM25,PM10,NO2'
AIR_start='2013-4-15_17:00:00'
AIR_length='1168'
AIR_freq='1H'
AIR_env='temperature,pressure,humidity,wind'

# btc and eth
# WITHOUT ENV
python gluonts/lzl_shared_ssm/models/prophet_compared/run_prophet.py -t=$target -st=$start -f=$freq -T=$length -pr=$pred -pa=$past -s=$slice &
# with ENV
for lags in 3 4 5 6 7
do
    for ((i=1;i<=1;i=i+1))#! repettion
    do
        python gluonts/lzl_shared_ssm/models/prophet_compared/run_prophet_env.py --target=$target --start=$start --timestep=$length --freq=$freq \
        --slice=$slice --past=$past --pred=$pred --env=$env --lags=$lags &
    done
done

# economic indice
# WITHOU ENV
# python gluonts/lzl_shared_ssm/models/prophet_compared/run_prophet.py -t=$E_target -st=$E_start -f=$E_freq -T=$E_length -pr=$pred -pa=$past -s=$slice &
# with ENV
# for lags in 3 4 5 6 7
# do
#     for ((i=1;i<=6;i=i+1))#! repettion
#     do
#         python gluonts/lzl_shared_ssm/models/prophet_compared/run_prophet_env.py --target=$E_target --start=$E_start --timestep=$E_length --freq=$E_freq \
#         --slice=$slice --past=$past --pred=$pred --env=$E_env --lags=$lags &
#     done
# done

# air quality
#WITHOUT ENV
# python gluonts/lzl_shared_ssm/models/prophet_compared/run_prophet.py -t=$AIR_target -st=$AIR_start -f=$AIR_freq -T=$AIR_length -pr=$pred -pa=$past -s=$slice &
# with ENV
# for lags in 1 2 3 4 5
# do
#     for ((i=1;i<=1;i=i+1))#! repettion
#     do
#         python gluonts/lzl_shared_ssm/models/prophet_compared/run_prophet_env.py --target=$AIR_target --start=$AIR_start --timestep=$AIR_length --freq=$AIR_freq \
#         --slice=$slice --past=$past --pred=$pred --env=$AIR_env --lags=$lags &
#     done
# done