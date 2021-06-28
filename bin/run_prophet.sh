#!/bin/bash
source activate gluonts
past='30'
pred='3'
slice='overlap'

# 比特币数据
target='btc,eth'
start='2018-08-02'
length='503'
freq='1D'
env='gold'


#金融指数数据
E_target="UKX,VIX,SPX,SHSZ300,NKY"
E_start='2008-01-04'
E_length='2606'
E_freq='1B'
E_env='SX5E,DXY'

# parser.add_argument('-t' ,'--target' , type=str , help='target series', default='btc,eth')
# parser.add_argument('-e' ,'--env' , type=str , help='enviroment as dynamic_feat_real', default='gold')
# parser.add_argument('-l' , '--lags' , type=int, help='featu_dynamic_real和目标序列之间的lag', default=5)
# parser.add_argument('-st','--start', type=str, help='start time of the dataset',default='2018-08-02')
# parser.add_argument('-f'  ,'--freq' , type = str , help = 'frequency of dataset' , default='1D')
# parser.add_argument('-T','--timestep', type=int, help='length of the dataset', default=503)
# parser.add_argument('-pr'  ,'--pred' , type = int , help = 'length of prediction range' , default=5)
# parser.add_argument('-pa'  ,'--past' , type = int , help = 'length of training range' , default=90)
# parser.add_argument('-s'  ,'--slice' , type = str , help = 'type of sliding dataset' , default='overlap')
#! 换行的\符号，千万不要在后面带空格

python gluonts/lzl_shared_ssm/models/prophet_compared/run_prophet.py -t=$target -st=$start -f=$freq -T=$length -pr=$pred -pa=$past -s=$slice &
# with ENV
for lags in 3 4 5 6 7
do
    for ((i=1;i<=;i=i+1))#! repettion
    do
        python gluonts/lzl_shared_ssm/models/prophet_compared/run_prophet_env.py --target=$target --start=$start --timestep=$length --freq=$freq \
        --slice=$slice --past=$past --pred=$pred --env=$env --lags=$lags &
    done
done


python gluonts/lzl_shared_ssm/models/prophet_compared/run_prophet.py -t=$E_target -st=$E_start -f=$E_freq -T=$E_length -pr=$pred -pa=$past -s=$slice &
# with ENV
for lags in 3 4 5 6 7
do
    for ((i=1;i<=6;i=i+1))#! repettion
    do
        python gluonts/lzl_shared_ssm/models/prophet_compared/run_prophet_env.py --target=$E_target --start=$E_start --timestep=$E_length --freq=$E_freq \
        --slice=$slice --past=$past --pred=$pred --env=$E_env --lags=$lags &
    done
done



