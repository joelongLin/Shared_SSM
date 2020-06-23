import numpy as np
import pandas as pd
import os

root='data_process/raw_data/'
ds_name = 'eth.csv'
time_str = ['beijing_time']

if __name__ == '__main__':
    path = os.path.join(root, ds_name)
    df = pd.read_csv(path , parse_dates=time_str , header = 0 ,index_col=time_str)
    df = df.fillna(method='pad')
    df_diff = df.diff()
    null_diff_index = np.where(pd.isnull(df_diff))

    df_diff.to_csv(os.path.join(root, ds_name.split('.')[0] + '_diff' + '.csv'))
    pass