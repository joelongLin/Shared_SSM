import pickle
import numpy as np

forecasts_path = 'data/predict_result/forecasts_electricity_{}_{}.pkl'.format(
                336
                ,168
            )
tss_path = 'data/predict_result/tss_electricity_{}_{}.pkl'.format(
                336
                ,168
            )
train_data_path = '../lzl_deepstate/data/train_electricity_336_168.pkl'
test_data_path = '../lzl_deepstate/data/test_electricity_504_168.pkl'
with open(test_data_path, 'rb') as fp:
    result = pickle.load(fp)
    print(type(result))
    pass
