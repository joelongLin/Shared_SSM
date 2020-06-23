# import json, urllib
# import urllib.request
#
# data = {}
# data["appkey"] = "718dbf5760ae0780"
# url_values = urllib.parse.urlencode(data)
# url = "https://api.binstd.com/gold/shgold" + "?" + url_values
# result = urllib.request.urlopen(url)
# jsonarr = json.loads(result.read())
#
# if jsonarr["status"] != u"0":
#     print(jsonarr["msg"])
#     exit()
# result = jsonarr["result"]
#
# for val in result:
#     print(val["type"], val["typename"], val["price"], val["openingprice"])
# import quandl
import numpy as np
import pandas as pd

# a  = quandl.get("LBMA/GOLD", authtoken="qhHzZjUqogcvGZPSVMLa")
a = pd.read_csv('raw_data/gold_lbma.csv',parse_dates=['Date'] ,index_col='Date')
np.where(a.isna() == True)
for i in range(1,a.shape[1]):
    print('{}列的缺失值数量为：{}'.format(a.columns[i]
                            , len(np.where(a.loc[:,a.columns[i]].isna() == True)[0])
                            ))

# print(a)