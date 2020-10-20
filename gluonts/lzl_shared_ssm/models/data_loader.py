# -*- coding: UTF-8 -*-
# author : joelonglin
import numpy as np
import random



# Standard library imports
import itertools
from collections import defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Optional  # noqa: F401

# First-party imports
from gluonts.core.component import DType
from gluonts.dataset.common import DataEntry, Dataset
from gluonts.transform import Transformation

# 完全参考gluonts 给定的API
DataBatch = Dict[str, Any]

# 完全参考gluonts,用于切分序列为的 past 以及 future
def _past(ts_field):
    return f"past_{ts_field}"
def _future(ts_field):
    return f"future_{ts_field}"
def _stack(value1 , value2 , dim):
    assert abs(len(value1.shape) - len(value2.shape)) <= 1 , "两者维度差距过大"
    if len(value1.shape) == len(value2.shape):
        return np.stack([value1, value2],axis=dim)
    elif len(value1.shape) > len(value2.shape):
        return np.concatenate([value1 , np.expand_dims(value2,axis=dim)] , axis=dim)
    else:
        return _stack(value2 ,value1 ,dim)

class BatchBuffer_NoMX:
    '''
    参考于 gluonts, 用于去除 mxnet 模块
    此类 封装的是 batch 的功能
    可以加入sample使其堆叠为batch
    '''
    def __init__(
        self, batch_size: int, dtype: DType = np.float32
    ) -> None:
        self._buffers: Dict[Any, List[Any]] = defaultdict(list)
        self.batch_size = batch_size
        self._size = 0
        self.dtype = dtype

    def add(self, d: Dict[str, List[np.ndarray]]):
        if self._buffers:
            assert self._buffers.keys() == d.keys()
        for k, v in d.items():
            self._buffers[k].append(v)
        self._size += 1

    def __len__(self):
        return self._size

    def next_batch(self) -> DataBatch:
        assert self._size > 0
        n = min(self._size, self.batch_size)
        batch = {k: self.stack(v[:n]) for k, v in self._buffers.items()}
        for key in self._buffers.keys():
            self._buffers[key] = self._buffers[key][n:]
        self._size -= n
        return batch

    def stack(self, xs):
        if isinstance(xs[0], np.ndarray):
            data = np.asarray(xs)
            if data.dtype.kind == "f":
                data = data.astype(self.dtype)
            return data
        elif isinstance(xs[0], list):
            return [self.stack(t) for t in zip(*[x for x in xs])]
        elif isinstance(xs[0], tuple):
            return tuple([self.stack(t) for t in zip(*[x for x in xs])])
        else:
            return xs  # stack all other types as list

    def shuffle(self):
        perm = np.random.permutation(self._size)
        for key in self._buffers.keys():
            li = self._buffers[key]
            self._buffers[key] = [li[i] for i in perm]

#TODO：序列都对应了一个Dataset, 而每个Dataset确对应不同的 DataLoader
def mergeIterOut(
        loaders : List[Iterable],
        fields : List[str],
        include_future : bool
):
    flag = True
    while flag:
        batch = {}
        for iter in loaders:
            try:
                i_batch = next(iter)
                if len(batch) == 0:
                    batch = i_batch.copy()
                else:
                    for name in fields:
                        _name = _past(name)
                        batch[_name] = np.concatenate((batch[_name], i_batch[_name]), axis=-1)
                        if include_future:
                            _name = _future(name)
                            batch[_name] = np.concatenate((batch[_name], i_batch[_name]), axis=-1)
            except Exception:
                flag = False
                batch = None
                break
            finally:
                pass
        yield batch

#将目标序列的单序列拼接起来
def stackIterOut(
        loaders : List[Iterable],
        fields : List[str] ,
        dim : int ,
        include_future : bool
):
    flag = True
    while flag:
        batch = {}

 
        for iter in loaders:
            
            
            try:
                i_batch = next(iter)
                
                if len(batch) == 0:
                    batch = i_batch.copy()                      
                else:
                    for name in fields:
                        _name = _past(name)
                        # expand dim to stack tensor
                        batch[_name] = _stack(batch[_name], i_batch[_name], dim=dim) 
                        if include_future:
                            _name = _future(name)
                            batch[_name] = _stack(batch[_name], i_batch[_name], dim=dim)
            except Exception:       
                flag = False
                batch = None
                break
            finally:
                pass
        yield batch


class DataLoader_NoMX(Iterable[DataEntry]):
    """
   一个来自于gluonts的DataLoader抽象类
   在这里去除其 Mxnet的部分

    Parameters
    ----------
    dataset
        The dataset from which to load data.
    transform
        A transformation to apply to each entry in the dataset.
    batch_size
        The size of the batches to emit.
    dtype
        Floating point type to use.
    """

    def __init__(
        self,
        dataset: Dataset,
        transform: Transformation,
        batch_size: int,
        dtype: DType = np.float32,
    ) -> None:
        self.dataset = dataset
        self.transform = transform
        self.batch_size = batch_size
        self.dtype = dtype

class TrainDataLoader_OnlyPast(DataLoader_NoMX):
       '''
        为Shared_SSM模型提供 训练时的数据

       因为数据预处理时已经切好片了，每条时间序列，所以is_train = False
       gluonts提供的的TrainDataLoader类因为 Dataset是大于等于 past+pred 的序列，因此具备了采样的操作
       而我们这里的数据集只有 past长度，且根本不需要采样。

       在这里如果 is_train = False 就意味着不需要采样，需要帮我们产生 future_time_feature

       Parameters
       ----------
       dataset
           The dataset from which to load data.
       transform
           A transformation to apply to each entry in the dataset.
       batch_size
           The size of the batches to emit.
       num_batches_per_epoch
           Number of batches to return in one complete iteration over this object.
       dtype
           Floating point type to use.
       '''

       def __init__(
               self,
               dataset: Dataset,
               transform: Transformation,
               batch_size: int,
               num_batches_per_epoch: int,
               dtype: DType = np.float32,
               shuffle_for_training: bool = False,
               num_batches_for_shuffling: int = 10,
       ) -> None:
           super().__init__(dataset, transform, batch_size, dtype)
           self.num_batches_per_epoch = num_batches_per_epoch
           self.shuffle_for_training = shuffle_for_training
           self._num_buffered_batches = (
               num_batches_for_shuffling if shuffle_for_training else 1
           )
           self._cur_iter: Optional[Iterator] = None
           self._buffer = BatchBuffer_NoMX(self.batch_size, dtype)

       def _emit_batches_while_buffer_larger_than(
               self, thresh
       ) -> Iterator[DataBatch]:
           if self.shuffle_for_training:
               self._buffer.shuffle()
           while len(self._buffer) > thresh:
               yield self._buffer.next_batch()

       def _iterate_forever(
               self, collection: Iterable[DataEntry]
       ) -> Iterator[DataEntry]:
           # iterate forever over the collection, the collection must be non empty
           while True:
               try:
                   # 这个 first 就是先经过 ProcessDataEntry 之后的数据集
                   first = next(iter(collection))
               except StopIteration:
                   raise Exception("empty dataset")
               else:
                   for x in itertools.chain([], collection):  # 在这里修改，使得代码不会重复遍历第一个 collection
                       yield x

       def __len__(self) -> int:
           return self.num_batches_per_epoch

       def __iter__(self) -> Iterator[DataBatch]:
           batch_count = 0
           if self._cur_iter is None:
               self._cur_iter = self.transform(
                   self._iterate_forever(self.dataset), is_train=False
               )  # 在使用 TrainDataLoader 的时候 会默认地使用 is_train=True
           assert self._cur_iter is not None
           while True:
               data_entry = next(self._cur_iter)
               self._buffer.add(data_entry)
               if (
                       len(self._buffer)
                       >= self._num_buffered_batches * self.batch_size  # 用于判断 是否 batch buffer 里面是否有足够多的 样本
               ):
                   for batch in self._emit_batches_while_buffer_larger_than(
                           self.batch_size - 1
                   ):
                       yield batch
                       batch_count += 1
                       # 这里面使用重定向 generator
                       if batch_count >= self.num_batches_per_epoch:
                           #就是因为这个没有置0 ，才导致后面的训练样本不全面
                           batch_count = 0
                           self._cur_iter = self.transform(
                               self._iterate_forever(self.dataset), is_train=False
                           )

class InferenceDataLoader_WithFuture(DataLoader_NoMX):
    """
    一个为Shared_SSM模型提供预测时数据的 DataLoader

    但是经过数据预处理时放入此InferenceLoader的数据集长度永远为 past+pred
    , 是包含预测域信息的, 所以此时我们不需要这个loader帮我们产生future 的 time_feature, 所以is_train=True

    在这里如果 is_train = True就意味着需要采样(只一次)，不需要帮我们产生 future_time_feature

    Parameters

    ----------
    dataset
        The dataset from which to load data.
    transform
        A transformation to apply to each entry in the dataset.
    batch_size
        The size of the batches to emit.
    ctx
        MXNet context to use to store data.
    dtype
        Floating point type to use.
    """

    def __iter__(self) -> Iterator[DataBatch]:
        buffer = BatchBuffer_NoMX(self.batch_size, self.dtype)
        # 这里面修改源码，只是想让loader只出一遍数据集，但是传入的数据集，其序列长度被固定为 past + pred
        for data_entry in self.transform(iter(self.dataset), is_train=True):
            buffer.add(data_entry)
            if len(buffer) >= self.batch_size:
                yield buffer.next_batch()
        if len(buffer) > 0:
            yield buffer.next_batch()

