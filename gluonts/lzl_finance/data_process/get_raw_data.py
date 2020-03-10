# -*- coding: UTF-8 -*-
# author : joelonglin
import datetime
import pandas as pd
import pandas_datareader.data as web

dataset_name = 'BTC-USD'

start = datetime.datetime(2016, 1, 1) # or start = '1/1/2016'
end = datetime.datetime(2020,1,9)
prices = web.DataReader(dataset_name, 'yahoo', start, end)

prices.to_csv('raw_data/btc_{}.csv'.format(str(start).split(' ')[0]))