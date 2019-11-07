import pickle
import numpy as np

forecast_result_path = ""
train_data_path = ''
with open('../lzl_deepstate/data/train_electricity_336_168.pkl', 'rb') as fp:
    result = pickle.load(fp)
    print(type(result))
    pass

    # print(all_train_data)


    # for elem in all_train_data:
    #     print(elem)