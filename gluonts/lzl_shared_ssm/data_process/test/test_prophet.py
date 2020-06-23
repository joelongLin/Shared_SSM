import pandas as pd
import os
os.chdir('/home/lzl/pycharm/gluon/gluonts/lzl_shared_ssm/')
from gluonts.model.prophet import ProphetPredictor
from gluonts.lzl_shared_ssm.utils import create_dataset_if_not_exist ,time_format_from_frequency_str
import pickle

target = 'btc,eth'
environment = 'gold'
maxlags=0
start = '2018-08-02'
freq = '1D'
timestep = 503
pred = 5
past = 90
slice = 'overlap'
name_prefix = 'data_process/processed_data/{}_{}_{}_{}.pkl'

if __name__ == '__main__':
    # 导入 target 以及 environment 的数据
    if slice == 'overlap':
        series = timestep - past - pred + 1
        print('每个数据集的序列数量为 ,', series)
    if slice == 'nolap':
        series = timestep // (past + pred)
        print('每个数据集的序列数量为 ,', series)
    # 目标序列的数据路径
    target_path = {name: name_prefix.format(
        '%s_start(%s)' % (name, start), '%s_DsSeries_%d' % (slice, series),
        'train_%d' % 90, 'pred_%d' % pred
    ) for name in target.split(',')}
    target_start = pd.Timestamp(start, freq=freq)
    env_start = target_start - maxlags * target_start.freq
    env_start = env_start.strftime(time_format_from_frequency_str(freq))
    print('environment 开始的时间：', env_start)
    env_path = {name: name_prefix.format(
        '%s_start(%s)' % (name, env_start), '%s_DsSeries_%d' % (slice, series),
        'train_%d' % 90, 'pred_%d' % pred,
    ) for name in environment.split(',')}

    create_dataset_if_not_exist(
        paths=target_path, start=start, past_length=past
        , pred_length=pred, slice=slice
        , timestep=timestep, freq=freq
    )

    create_dataset_if_not_exist(
        paths=env_path, start=env_start, past_length=past
        , pred_length=pred, slice=slice
        , timestep=timestep, freq=freq
    )

    target_data, env_data = [], []
    # 由于 target 应该都是 dim = 1 只是确定有多少个 SSM 而已
    for target_name in target_path:
        target = target_path[target_name]
        with open(target, 'rb') as fp:
            target_ds = pickle.load(fp)
            assert target_ds.metadata['dim'] == 1, 'target 序列的维度都应该为1'
            target_data.append(target_ds)
    env_dim = 0
    for env_name in env_path:
        env = env_path[env_name]
        with open(env, 'rb') as fp:
            env_ds = pickle.load(fp)
            env_dim += env_ds.metadata['dim']
            env_data.append(env_ds)
    print('导入原始数据成功~~~')

    prophet_predictor = ProphetPredictor(freq=freq, prediction_length=pred)
    # btc
    generators = prophet_predictor.predict(target_data[0].train)
    sample_forecasts = list(generators)

    import pickle
    with open('./sample_forecasts.pkl' , 'wb') as fp:
        pickle.dump(sample_forecasts , fp)