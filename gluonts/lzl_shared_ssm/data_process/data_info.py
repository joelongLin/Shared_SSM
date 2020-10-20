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
    "btc_diff": DatasetInfo(
        name="btc_diff",
        url=root + 'btc_diff.csv',
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
    "eth_diff": DatasetInfo(
        name="eth_diff",
        url=root + 'eth_diff.csv',
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
    "gold_diff": DatasetInfo(
        name="gold_diff",
        url=root + 'GOLD_diff.csv',
        time_col='Date',
        dim=2,
        aim=['Open','Close'],
    ),
    "gold_lbma": DatasetInfo(
        name="gold_lbma",
        url=root + 'gold_lbma.csv',
        time_col='Date',
        dim=2,
        aim=['USD(AM)', 'GBP(AM)'],
    ),
    "gold_lbma_diff": DatasetInfo(
        name="gold_lbma_diff",
        url=root + 'gold_lbma_diff.csv',
        time_col='Date',
        dim=2,
        aim=['USD(AM)', 'GBP(AM)'],
    ),
    "LMEAluminium": DatasetInfo(
        name="Aluminium",
        url=[
            root + 'SIGIR/Train/Train_data/LMEAluminium3M_train.csv' , 
            root + 'SIGIR/Validation/Validation_data/LMEAluminium3M_validation.csv',
            root + 'SIGIR/Test/Test_data/LMEAluminium3M_test.csv'
        ],
        time_col='Unnamed: 0.1',
        index_col=0,
        dim=1,
        aim=['Close.Price'],
    ),
    "LMECopper": DatasetInfo(
        name="Copper",
        url= [
            root + 'SIGIR/Train/Train_data/LMECopper3M_train.csv', 
            root + 'SIGIR/Validation/Validation_data/LMECopper3M_validation.csv',
            root + 'SIGIR/Test/Test_data/LMECopper3M_test.csv'
        ],
        time_col='Unnamed: 0.1',
        index_col=0,
        dim=1,
        aim=['Close.Price'],
    ),
    "LMELead": DatasetInfo(
        name="Lead",
        url=[
            root + 'SIGIR/Train/Train_data/LMELead3M_train.csv', 
            root + 'SIGIR/Validation/Validation_data/LMELead3M_validation.csv',
            root + 'SIGIR/Test/Test_data/LMELead3M_test.csv'
        ],
        time_col='Unnamed: 0.1',
        index_col=0,
        dim=1,
        aim=['Close.Price'],
    ),
    "LMENickel": DatasetInfo(
        name="Nickel",
        url=[
            root + 'SIGIR/Train/Train_data/LMENickel3M_train.csv',
            root +'SIGIR/Validation/Validation_data/LMENickel3M_validation.csv',
            root +'SIGIR/Test/Test_data/LMENickel3M_test.csv'
        ],
        time_col='Unnamed: 0.1',
        index_col=0,
        dim=1,
        aim=['Close.Price'],
    ),
    "LMETin": DatasetInfo(
        name="Tin",
        url=[
            root + 'SIGIR/Train/Train_data/LMETin3M_train.csv' , 
            root + 'SIGIR/Validation/Validation_data/LMETin3M_validation.csv',
            root + 'SIGIR/Test/Test_data/LMETin3M_test.csv',
        ],
        time_col='Unnamed: 0.1',
        index_col=0,
        dim=1,
        aim=['Close.Price'],
    ),
    "LMEZinc": DatasetInfo(
        name="Zinc",
        url=[
            root + 'SIGIR/Train/Train_data/LMEZinc3M_train.csv' , 
            root + 'SIGIR/Validation/Validation_data/LMEZinc3M_validation.csv',
            root + 'SIGIR/Test/Test_data/LMEZinc3M_test.csv'
        ],
        time_col='Unnamed: 0.1',
        index_col=0,
        dim=1,
        aim=['Close.Price'],
    ),
        
    "comexCopper": DatasetInfo(
        name="COMEX_Copper",
        url=[
            root + 'SIGIR/Train/Train_data/COMEX_Copper_train.csv' , 
            root + 'SIGIR/Validation/Validation_data/COMEX_Copper_validation.csv',
            root + 'SIGIR/Test/Test_data/COMEX_Copper_test.csv'
        ],
        time_col='Unnamed: 0.1',
        index_col=0,
        dim=1,
        aim=['Close'],
    ),
    "comexGold": DatasetInfo(
        name="COMEX_Gold",
        url=[
            root + 'SIGIR/Train/Train_data/COMEX_Gold_train.csv' , 
            root + 'SIGIR/Validation/Validation_data/COMEX_Gold_validation.csv',
            root + 'SIGIR/Test/Test_data/COMEX_Gold_test.csv'
        ],
        time_col='Unnamed: 0.1',
        index_col=0,
        dim=1,
        aim=['Close'],

    ),
    "comexPalladium":DatasetInfo(
        name="COMEX_Palladium",
        url=[
            root + 'SIGIR/Train/Train_data/COMEX_Palladium_train.csv' , 
            root + 'SIGIR/Validation/Validation_data/COMEX_Palladium_validation.csv',
            root + 'SIGIR/Test/Test_data/COMEX_Palladium_test.csv'
        ],
        time_col='Unnamed: 0.1',
        index_col=0,
        dim=1,
        aim=['Close'],
    ),
    "comexPlatinum":DatasetInfo(
        name="COMEX_Platinum",
        url=[
            root + 'SIGIR/Train/Train_data/COMEX_Platinum_train.csv' , 
            root + 'SIGIR/Validation/Validation_data/COMEX_Platinum_validation.csv',
            root + 'SIGIR/Test/Test_data/COMEX_Platinum_test.csv'
        ],
        time_col='Unnamed: 0.1',
        index_col=0,
        dim=1,
        aim=['Close'],
    ),
    "comexSilver":DatasetInfo(
        name="COMEX_Silver",
        url=[
            root + 'SIGIR/Train/Train_data/COMEX_Silver_train.csv' , 
            root + 'SIGIR/Validation/Validation_data/COMEX_Silver_validation.csv',
            root + 'SIGIR/Test/Test_data/COMEX_Silver_test.csv'
        ],
        time_col='Unnamed: 0.1',
        index_col=0,
        dim=1,
        aim=['Close'],
    ),
    # Indices_NKY Index_train.csv
    # Indices_SHSZ300 Index_train.csv
    # Indices_SPX Index_train.csv
    # Indices_SX5E Index_train.csv
    # Indices_UKX Index_train.csv
    # Indices_VIX Index_train.csv
    "DXY":DatasetInfo(
        name="DXY",
        url=root + 'SIGIR/Train/Train_data/Indices_DXY Curncy_train.csv' , 
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
    
}