from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
from typing import Sequence, Any

import numpy as np
from pandas import DataFrame
from prettytable import PrettyTable
from torch.utils.data import Dataset, Subset

from utils.data.partitioner import DataPartitioner, get_sample_num, get_target


class FederatedDataset(ABC):

    @abstractmethod
    def __contains__(self, key):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abstractmethod
    def train(self, key) -> Dataset:
        raise NotImplementedError

    @abstractmethod
    def val(self, key) -> Dataset:
        raise NotImplementedError

    @abstractmethod
    def test(self) -> Dataset:
        raise NotImplementedError


class DPFederatedDataset(FederatedDataset):

    def __init__(self, dataset: Dataset, dp: DataPartitioner):
        self._dp = dp
        self._dataset = dataset

    def __contains__(self, key):
        return key in self._dp

    def __len__(self):
        return len(self._dp)

    def __iter__(self):
        for k in self._dp:
            yield k

    def train(self, key) -> Dataset:
        return Subset(self._dataset, self._dp.train_indices[key])

    def val(self, key) -> Dataset:
        return Subset(self._dataset, self._dp.val_indices[key])

    def test(self) -> Dataset:
        return Subset(self._dataset, self._dp.test_indices)


class DictFederatedDataset(FederatedDataset):

    def __init__(self, train: dict[Any, Dataset], val: dict[Any, Dataset], test: Dataset):
        self._train, self._val, self._test = train, val, test

    def __contains__(self, key):
        return key in self._train

    def __len__(self):
        return len(self._train)

    def __iter__(self):
        for k in self._train:
            yield k

    def train(self, key) -> Dataset:
        return self._train[key]

    def val(self, key) -> Dataset:
        return self._val[key]

    def test(self) -> Dataset:
        return self._test


class Summary:

    @staticmethod
    def stats(labels: Sequence):
        return OrderedDict(sorted(Counter(labels).items(), key=lambda x: x[0]))

    def __init__(self, fds: FederatedDataset):
        self.fds = fds
        self.device_num = len(self.fds)
        self.labels = np.unique(get_target(self.fds.test()))
        self.train_sample_num = sum(map(lambda x: get_sample_num(fds.train(x)), fds))
        self.val_sample_num = sum(map(lambda x: get_sample_num(fds.val(x)), fds))
        self.test_sample_num = get_sample_num(fds.test())

    def train(self):
        df = DataFrame(
            np.zeros((len(self.fds), self.labels.shape[0])),
            index=list(self.fds), columns=self.labels, dtype=int
        )
        for cid in self.fds:
            for k, v in self.stats(get_target(self.fds.train(cid))).items():
                df.loc[cid, k] = v
        return df

    def val(self):
        df = DataFrame(
            np.zeros((len(self.fds), self.labels.shape[0])),
            index=list(self.fds), columns=self.labels, dtype=int
        )
        for cid in self.fds:
            for k, v in self.stats(get_target(self.fds.val(cid))).items():
                df.loc[cid, k] = v
        return df

    def test(self):
        df = DataFrame(self.stats(get_target(self.fds.test())), index=[''])
        return df

    def __str__(self):
        # 创建一个PrettyTable对象，并设置默认有垂直和水平分割线
        table = PrettyTable()
        # 添加表头
        table.field_names = ["Dataset", self.fds.__class__.__name__]

        # 添加数据集基本信息
        table.add_row(["Number of clients", self.device_num])
        table.add_row(["Number of train samples", self.train_sample_num])
        table.add_row(["Number of val samples", self.val_sample_num])
        table.add_row(["Number of test samples", self.test_sample_num])

        # 对于train/val/test()返回的不可直接转换为表格形式的内容，保持文本描述
        table.add_row(["Train Distribution", str(self.train())])
        table.add_row(["Val Distribution", str(self.val())])
        table.add_row(["Test Distribution", str(self.test())])

        # 返回格式化后的表格字符串
        return table.get_string()

    def to_csv(self, path):
        self.train().to_csv(path + 'train.csv')
        self.val().to_csv(path + 'val.csv')
        self.test().to_csv(path + 'test.csv')


def summary(fds: FederatedDataset):
    fdss = Summary(fds)
    print(fdss)
