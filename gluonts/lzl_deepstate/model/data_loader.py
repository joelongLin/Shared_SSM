import numpy as np
import pandas as pd
import random

class DataLoader(object):

    # 输入一个 list[samples] ,其中每个 sample 都是一个list
    def __init__(self,list_dataset,config):
        self.list_dataset = list_dataset
        self.batch_size = config.batch_size
        self.num_batch =  config.num_batches_per_epoch
        self.shuffle = config.shuffle

    def data_iterator(self):
        order = list(range(len(self.list_dataset)))

        if self.shuffle:
            random.seed(230)
            random.shuffle(order)

        start = 0
        end = start + self.batch_size
        while True:  # 改到这里
            start = start % (len(self.list_dataset))
            end = end % (len(self.list_dataset))
            net_input = []
            # fetch sentences and tags
            batch_train = [self.list_dataset[idx] for idx in order[start:end]]\
                if start < end else [self.list_dataset[idx] for idx in order[start:]+order[:end]]
            # 对数据进行重新改造，将每一个数据进行stack
            for input_no in range(len(batch_train[0])):
                input = [batch_train[sample_no][input_no] for sample_no in range(len(batch_train))]
                input = np.stack(input,axis=0)
                net_input.append(input)

            start = end
            end = start + self.batch_size

            yield net_input

