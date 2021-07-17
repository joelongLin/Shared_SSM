from typing import List, NamedTuple, Optional , Union

root='data_process/raw_data/'
class DatasetInfo(NamedTuple):
    name: str  # 该数据集的名称
    url: Union[str, List[str]]  # 存放该数据集的路径
    time_col: str  # 表明这些 url 里面对应的 表示时刻的名称 如: eth.csv 中的 beijing_time  或者 btc.csv 中的Date
    dim: int  # 该数据集存在多少条序列
    aim: List[str]  # 序列的目标特征，包含的数量特征应该与 url， time_col一致
    index_col: Optional[int] = None  # 是否存在多余的一列 index
    feat_dynamic_cat: Optional[str] = None  # 表明该序列类别的 (seq_length , category)

datasets_info = {
    "btc": DatasetInfo(
        name="btc",
        url=root + 'btc.csv',
        time_col='beijing_time',
        dim=1,
        aim=['close'],
    ),
    "eth": DatasetInfo(
        name="eth",
        url=root + 'eth.csv',
        time_col='beijing_time',
        dim=1,
        aim=['close'],  # 这样子写只是为了测试预处理程序是否强大
    ),
    "gold": DatasetInfo(
        name="gold",
        url=root + 'Barrick-GOLD.csv',
        time_col='Date',
        dim=2,
        aim=['Open','Close'],
    ),
    # Indices_NKY Index_train.csv
    # Indices_SHSZ300 Index_train.csv
    # Indices_SPX Index_train.csv
    # Indices_SX5E Index_train.csv
    # Indices_UKX Index_train.csv
    # Indices_VIX Index_train.csv
    "DXY":DatasetInfo(
        name="DXY",
        url=root + 'SIGIR/Train/Train_data/Indices_DXY Index_train.csv' , 
        time_col='Unnamed: 0.1',
        index_col=0,
        dim=1,
        aim=['DXY'],
    ),
    "NKY":DatasetInfo(
        name="NKY",
        url=root + 'SIGIR/Train/Train_data/Indices_NKY Index_train.csv' , 
        time_col='Unnamed: 0.1',
        index_col=0,
        dim=1,
        aim=['NKY'],
    ),
    "SHSZ300":DatasetInfo(
        name="SHSZ300",
        url=root + 'SIGIR/Train/Train_data/Indices_SHSZ300 Index_train.csv' , 
        time_col='Unnamed: 0.1',
        index_col=0,
        dim=1,
        aim=['SHSZ300'],
    ),
    "SPX":DatasetInfo(
        name="SPX",
        url=root + 'SIGIR/Train/Train_data/Indices_SPX Index_train.csv' , 
        time_col='Unnamed: 0.1',
        index_col=0,
        dim=1,
        aim=['SPX'],
    ),
    "SX5E":DatasetInfo(
        name="SX5E",
        url=root + 'SIGIR/Train/Train_data/Indices_SX5E Index_train.csv' , 
        time_col='Unnamed: 0.1',
        index_col=0,
        dim=1,
        aim=['SX5E'],
    ),
    "UKX":DatasetInfo(
        name="UKX",
        url=root + 'SIGIR/Train/Train_data/Indices_UKX Index_train.csv' , 
        time_col='Unnamed: 0.1',
        index_col=0,
        dim=1,
        aim=['UKX'],
    ),
    "VIX":DatasetInfo(
        name="VIX",
        url=root + 'SIGIR/Train/Train_data/Indices_VIX Index_train.csv' , 
        time_col='Unnamed: 0.1',
        index_col=0,
        dim=1,
        aim=['VIX'],
    ),
    "PM25":DatasetInfo(
        name="PM25",
        url=root + 'SH_weather.txt' , 
        time_col='time',
        index_col=None,
        dim=1,
        aim=['PM25_AQI_value'],
    ),
    "PM10":DatasetInfo(
        name="PM10",
        url=root + 'SH_weather.txt' , 
        time_col='time',
        index_col=None,
        dim=1,
        aim=['PM10_AQI_value'],
    ),
    "NO2":DatasetInfo(
        name="NO2",
        url=root + 'SH_weather.txt' , 
        time_col='time',
        index_col=None,
        dim=1,
        aim=['NO2_AQI_value'],
    ),
    "temperature":DatasetInfo(
        name="temperature",
        url=root + 'SH_weather.txt' , 
        time_col='time',
        index_col=None,
        dim=1,
        aim=['temperature'],
    ),
    "pressure":DatasetInfo(
        name="pressure",
        url=root + 'SH_weather.txt' , 
        time_col='time',
        index_col=None,
        dim=1,
        aim=['pressure'],
    ),
    "humidity":DatasetInfo(
        name="humidity",
        url=root + 'SH_weather.txt' , 
        time_col='time',
        index_col=None,
        dim=1,
        aim=['humidity'],
    ),
    "wind":DatasetInfo(
        name="wind",
        url=root + 'SH_weather.txt' , 
        time_col='time',
        index_col=None,
        dim=1,
        aim=['wind'],
    ),

}