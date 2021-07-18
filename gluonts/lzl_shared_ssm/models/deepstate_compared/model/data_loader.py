import numpy as np
import pandas as pd
import random
from typing import Dict, Iterator, NamedTuple, Optional, Tuple, Union,Any
DataEntry = Dict[str, Any]
import pickle
from gluonts.dataset.repository.datasets import get_dataset

class GroundTruthLoader(object):
    def __init__(self, config):
        self.config = config
        try:
            self.ds = get_dataset(config.dataset, regenerate=False)
        except:
            print('get ground truth from outside')
            with open('data/groundtruth_{}_{}_{}.pkl'.format(
            config.dataset, config.past_length, config.prediction_length,
            ), 'rb') as fp:
                self.ds = pickle.load(fp)

    def add_ts_dataframe(self,data_iterator: Iterator[DataEntry]):
        ts_df = []
        for data_entry in data_iterator:
            data = data_entry.copy()
            index = pd.date_range(
                    start=data["start"],
                freq=self.config.freq,
                periods=data["target"].shape[-1],
            )
            data["ts"] = pd.DataFrame(
                index=index, data=data["target"].transpose()
            )
            ts_df.append(data)

        return ts_df

    def get_ground_truth(self) :
        ground_truth = []
        for data_entry in self.add_ts_dataframe(iter(self.ds.test)):
            ground_truth.append(data_entry["ts"])
        return ground_truth

class DataLoader(object):

    
    def __init__(self,list_dataset,config):
        self.list_dataset = list_dataset
        self.batch_size = config.batch_size
        self.num_batch =  config.num_batches_per_epoch
        self.shuffle = config.shuffle

    def data_iterator(self, is_train):
        order = list(range(len(self.list_dataset)))

        if self.shuffle:
            random.seed(230)
            random.shuffle(order)

        start = 0
        end = start + self.batch_size
        flag = end > start
        while is_train or flag:  
            start = start % (len(self.list_dataset))
            end = end % (len(self.list_dataset))
            flag = end > start

            net_input = []
            net_input_other_info = []
            # fetch sentences and tags
            batch_train = [self.list_dataset[idx] for idx in order[start:end]]\
                if start < end else [self.list_dataset[idx] for idx in order[start:]+order[:end]]

            
            for end_input in range(len(batch_train[0])-1,0,-1):
                if isinstance(batch_train[0][end_input] , np.ndarray):
                    break
                    
            
            for input_no in range(end_input+1): 
                input = [batch_train[sample_no][input_no]
                         for sample_no in range(len(batch_train))]
                input = np.stack(input,axis=0)
                net_input.append(input)

            if end_input != len(batch_train[0])-1:
                for i in range(len(batch_train[0])-1,end_input,-1): 
                    other_info = [batch_train[sample_no][i]
                         for sample_no in range(len(batch_train))]
                    net_input_other_info.append(other_info)

            start = end
            end = start + self.batch_size

            yield net_input , net_input_other_info

