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

DataEntry = Dict[str, Any]
DataBatch = Dict[str, Any]

def _past(ts_field):
    return f"past_{ts_field}"


def _future(ts_field):
    return f"future_{ts_field}"

class BatchBuffer_NoMX:
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
                        batch[_name] = np.stack((batch[_name], i_batch[_name]), axis=dim)
                        if include_future:
                            _name = _future(name)
                            batch[_name] = np.stack((batch[_name], i_batch[_name]), axis=dim)
            except Exception:
                flag = False
                batch = None
                break
            finally:
                pass
        yield batch

class DataLoader_NoMX(Iterable[DataEntry]):
    """
    An abstract Iterable type for iterating and transforming a dataset,
    in batches of a prescribed size.

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

# 因为数据预处理时已经切好片了，每条时间序列，所以is_train = False
class TrainDataLoader_OnlyPast(DataLoader_NoMX):
           """
           An Iterable type for iterating and transforming a dataset, in batches of a
           prescribed size, until a given number of batches is reached.

           The transformation are applied with in training mode, i.e. with the flag
           `is_train = True`.

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
           """

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
                           if batch_count >= self.num_batches_per_epoch:
                               return

# 因为数据预处理时放入此InferenceLoader的数据集长度永远为 past+pred, 是包含预测域信息的, 所以is_train=True
class InferenceDataLoader_WithFuture(DataLoader_NoMX):
    """
    An Iterable type for iterating and transforming a dataset just once, in
    batches of a prescribed size.

    The transformation are applied with in inference mode, i.e. with the flag
    `is_train = False`. 不能无限产生数据，且产生数据的模式 is_train=False

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
        # TODO: 这里面修改源码，只是想让loader只出一遍数据集，但是传入的数据集 是包括了 past + pred
        for data_entry in self.transform(iter(self.dataset), is_train=True):
            buffer.add(data_entry)
            if len(buffer) >= self.batch_size:
                yield buffer.next_batch()
        if len(buffer) > 0:
            yield buffer.next_batch()

class DataLoader(object):
    # 输入一个 list[samples] ,其中每个 sample 都是一个 封装好的 TrainDataset
    def __init__(self,dataset,config):
        self.dataset = dataset
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
        while is_train or flag:  # 训练模式，无限循环提取数据
            start = start % (len(self.list_dataset))
            end = end % (len(self.list_dataset))
            flag = end > start

            net_input = []
            net_input_other_info = []
            # fetch sentences and tags
            batch_train = [self.list_dataset[idx] for idx in order[start:end]]\
                if start < end else [self.list_dataset[idx] for idx in order[start:]+order[:end]]

            # 获得不应该为输入网络的信息的初始位置
            for end_input in range(len(batch_train[0])-1,0,-1):
                if isinstance(batch_train[0][end_input] , np.ndarray):
                    # print('当前第%d个input位，确实是网络所需要的数据' % end_input)
                    break
                    # print('当前第%d个input位，不是网络所需要的数据' % end_input)
            # 对数据进行重新改造，将每一个数据进行stack
            for input_no in range(end_input+1): #你的遍历是要包括 endpoint的
                input = [batch_train[sample_no][input_no]
                         for sample_no in range(len(batch_train))]
                input = np.stack(input,axis=0)
                net_input.append(input)

            if end_input != len(batch_train[0])-1:
                for i in range(len(batch_train[0])-1,end_input,-1): #去除的其他信息不包括 endpoint
                    other_info = [batch_train[sample_no][i]
                         for sample_no in range(len(batch_train))]
                    net_input_other_info.append(other_info)

            start = end
            end = start + self.batch_size

            yield net_input , net_input_other_info

