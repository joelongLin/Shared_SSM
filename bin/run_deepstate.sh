#!/bin/bash
#source activate gluonts
#frozen

source activate tf_mx
past='30'
pred='3'
slice='overlap'
batch_size='32'
epoch='100'

# 比特币数据
#(90,5,13) (60,3,14) (30,1,15)
target='btc,eth'
start='2018-08-02'
length='503'
freq='1D'
env='gold'
batch_num='15'

#金融指数数据
# (30,1,81) (60,3,80) (90,5,79) batch_size = 32  --> 2606
# (30,1,89) (60,3,88) (90,5,87) batch_size = 32  --> 2867
E_target="UKX,VIX,SPX,SHSZ300,NKY"
E_start='2008-01-04'
E_length='2606'
E_freq='1B'
E_env='SX5E,DXY'
E_batch_num='81'

# parser.add_argument('-t' ,'--target' , type=str , help='target series', default='btc,eth')
# parser.add_argument('-e' ,'--env' , type=str , help='enviroment as dynamic_feat_real', default='gold')
# parser.add_argument('-l' , '--lags' , type=int, help='featu_dynamic_real和目标序列之间的lag', default=5)
# parser.add_argument('-st','--start', type=str, help='start time of the dataset',default='2018-08-02')
# parser.add_argument('-f'  ,'--freq' , type = str , help = 'frequency of dataset' , default='1D')
# parser.add_argument('-T','--timestep', type=int, help='length of the dataset', default=503)
# parser.add_argument('-pr'  ,'--pred' , type = int , help = 'length of prediction range' , default=5)
# parser.add_argument('-pa'  ,'--past' , type = int , help = 'length of training range' , default=90)
# parser.add_argument('-s'  ,'--slice' , type = str , help = 'type of sliding dataset' , default='overlap')
# parser.add_argument('-bs'  ,'--batch_size' , type = int , help = 'batch size' , default=32)
# parser.add_argument('-bn'  ,'--batch_num' , type = int , help = 'batch number per epoch' , default=15)
# parser.add_argument('-ep'  ,'--epochs' , type = int , help = 'epochs to train' , default=1)
#! 换行的\符号，千万不要在后面带空格

# without ENV
python gluonts/lzl_shared_ssm/models/deepstate_compared/run_deepstate_gluonts.py --target=$E_target --start=$E_start --timestep=$E_length --freq=$E_freq \
        --slice=$slice --batch_size=$batch_size --batch_num=$E_batch_num --epochs=$epoch --past=$past --pred=$pred

# with ENV
for lags in 3 4 5 6 7
do
    for ((i=1;i<=1;i=i+1))#! repettion
    do
        python gluonts/lzl_shared_ssm/models/deepstate_compared/run_deepstate_gluonts_env.py --target=$target --start=$start --timestep=$length --freq=$freq \
        --slice=$slice --batch_size=$batch_size --batch_num=$batch_num --epochs=$epoch --past=$past --pred=$pred --env=$env --lags=$lags
    done
done

# for lags in 5 6 7 8 9
# do
#     for ((i=1;i<=1;i=i+1))#! repettion
#     do
#         python gluonts/lzl_shared_ssm/models/deepstate_compared/run_deepstate_gluonts_env.py --target=$E_target --start=$E_start --timestep=$E_length --freq=$E_freq \
#         --slice=$slice --batch_size=$batch_size --batch_num=$E_batch_num --epochs=$epoch --past=$past --pred=$pred --env=$E_env --lags=$lags
#     done
# done

#! 用于运行普通 deepstate 的情况

