from enum import Enum
from collections import Counter
from abc import ABC, abstractmethod

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

from config.env import SEED, WORKER_NUM, PREFETCH_FACTOR
from utils.format import print_df
from utils.tool import set_seed
import utils.data.functional as F


def train_val_test_split(array, val: float, test: float, seed=None, shuffle=True):
    train, val_test = train_test_split(
        array,
        test_size=test + val,
        random_state=seed,
        shuffle=shuffle
    )
    val, test = train_test_split(
        array,
        test_size=test / (test + val),
        random_state=seed,
        shuffle=shuffle
    )
    return train, val, test


def get_target(dataset: Dataset):
    return np.concatenate(list(map(
        lambda x: x[-1].numpy(),
        DataLoader(dataset, num_workers=WORKER_NUM * 2, prefetch_factor=PREFETCH_FACTOR * 4, batch_size=1024)
    )))


def get_data(dataset: Dataset):
    return np.concatenate(list(map(
        lambda x: x[0].numpy(),
        DataLoader(dataset, num_workers=WORKER_NUM * 2, prefetch_factor=PREFETCH_FACTOR, batch_size=1024)
    )))


def get_sample_num(dataset: Dataset):
    return len(dataset)


def sample_by_class(dataset: Dataset, num_classes: int, size: int, seed=None):
    set_seed(seed, has_torch=True)
    samples, counter = {}, 0
    data, target = [], []
    for x, y in DataLoader(dataset, shuffle=True, batch_size=1):
        if samples.get(y.item(), 0) < size:
            data.append(x)
            target.append(y)
            samples[y.item()] = samples.get(y.item(), 0) + 1
            counter += 1
        if counter == num_classes * size:
            break
    return TensorDataset(torch.cat(data), torch.cat(target))


class Part(Enum):
    IID_BALANCE = 0
    IID_UNBALANCE_LOGNORMAL = 1
    IID_UNBALANCE_DIRICHLET = 2
    NONIID_BALANCE_CLIENT_DIRICHLET = 3
    NONIID_UNBALANCE_LOGNORMAL_CLIENT_DIRICHLET = 4
    NONIID_UNBALANCE_DIRICHLET_CLIENT_DIRICHLET = 5
    NONIID_LABEL_SKEW = 6
    NONIID_SHARD = 7
    NONIID_DIRICHLET = 8
    NONIID_SYNTHETIC = 9


class DataPartitioner(ABC):
    """
        Base class for data partition in federated learning.
    """

    def __init__(self):
        self.indices = {}
        self.train_indices = {}
        self.val_indices = {}
        self.test_indices = []

    @abstractmethod
    def _split(self):
        raise NotImplementedError

    @abstractmethod
    def __contains__(self, item):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError


class BasicPartitioner(DataPartitioner):

    def __init__(self, targets, args: dict):
        super(BasicPartitioner, self).__init__()
        self.args = args
        self.targets = targets if isinstance(targets, np.ndarray) else np.array(targets)
        self.sample_num = self.targets.shape[0]
        self.class_num = np.unique(self.targets).shape[0]
        self.part = Part(self.args.get('part', 0))
        self.client_num = args.get('client_num', 100)
        self.seed = self.args.get('seed', SEED)
        self.indices |= self._split()
        self._local_split(self.args.get('val', 0.2), self.args.get('test', 0.1))

    def _local_split(self, val, test):
        for i in self.indices:
            T, v, t = train_val_test_split(self.indices[i], val, test, seed=self.seed)
            self.train_indices[i], self.val_indices[i], test_indices = T.tolist(), v.tolist(), t.tolist()
            self.test_indices.extend(test_indices)

    def __len__(self):
        return len(self.indices)

    def __contains__(self, index):
        return index in self.indices

    def __iter__(self):
        for idx in self.indices:
            yield idx

    def __getitem__(self, index):
        return self.indices[index]

    def _split(self):
        # 切分数据集
        set_seed(self.seed)
        if self.part == Part.IID_BALANCE:
            return F.iid_partition(
                F.balance_split(self.client_num, self.sample_num),
                self.sample_num
            )
        elif self.part == Part.IID_UNBALANCE_LOGNORMAL:
            return F.iid_partition(
                F.unbalance_lognormal_split(self.client_num, self.sample_num, self.args.get('lognormal_sgm', 0.1)),
                self.sample_num
            )
        elif self.part == Part.IID_UNBALANCE_DIRICHLET:
            return F.iid_partition(
                F.unbalance_dirichlet_split(self.client_num, self.sample_num, self.args.get('dirichlet_alpha', 0.4),
                                            self.args.get('dirichlet_min', 2)),
                self.sample_num
            )
        if self.part == Part.NONIID_BALANCE_CLIENT_DIRICHLET:
            return F.noniid_client_dirichlet_partition(
                F.balance_split(self.client_num, self.sample_num),
                self.targets, self.class_num, self.args.get('dirichlet_alpha', 0.4)
            )
        elif self.part == Part.NONIID_UNBALANCE_LOGNORMAL_CLIENT_DIRICHLET:
            return F.noniid_client_dirichlet_partition(
                F.unbalance_lognormal_split(self.client_num, self.sample_num, self.args.get('lognormal_sgm', 0.1)),
                self.targets, self.class_num, self.args.get('dirichlet_alpha', 0.4)
            )
        elif self.part == Part.NONIID_UNBALANCE_DIRICHLET_CLIENT_DIRICHLET:
            return F.noniid_client_dirichlet_partition(
                F.unbalance_dirichlet_split(self.client_num, self.sample_num, self.args.get('dirichlet_alpha', 0.4),
                                            self.args.get('dirichlet_min', 2)),
                self.targets, self.class_num, self.args.get('dirichlet_alpha', 0.4)
            )
        elif self.part == Part.NONIID_LABEL_SKEW:
            # label-distribution-skew:quantity-based
            assert self.args.get('major_class_num', 3) <= self.class_num, f"major_class_num >= {self.class_num}"
            return F.noniid_label_skew_quantity_based_partition(
                self.targets, self.client_num, self.class_num, self.args.get('major_class_num', 3)
            )
        elif self.part == Part.NONIID_SHARD:
            # partition is 'shards'
            assert self.args.get('shard_num', 2) >= self.client_num, f"shard_num >= {self.client_num}"
            return F.noniid_shard_partition(
                self.targets, self.client_num, self.args.get('shard_num', 2)
            )
        elif self.part == Part.NONIID_DIRICHLET:
            # label-distribution-skew:distributed-based (Dirichlet)
            return F.noniid_dirichlet_partition(
                self.targets, self.client_num, self.class_num, self.args.get('dirichlet_alpha', 0.4),
                self.args.get('dirichlet_min', 2)
            )
        else:
            raise NotImplementedError("{} is supported !".format(self.part))

    def __str__(self):
        return get_report(self.targets, self.indices).head(10).to_string()


def get_report(targets, client_indices, verbose=False):
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)
    class_num = np.unique(targets).shape[0]
    columns, data = ['client_id', 'sample_num'], []
    columns.extend('class{}'.format(i) for i in range(class_num))
    for c_id in sorted(client_indices.keys()):
        c_targets, c_sample_num = targets[client_indices[c_id]], len(client_indices[c_id])
        c_target_cnt = Counter(c_targets)
        row = [c_id, c_sample_num]
        row.extend([c_target_cnt[i] for i in range(class_num)])
        data.append(row)
    df = DataFrame(data, columns=columns)
    if verbose is True:
        print_df(df)
    return df


def barh_report(report: DataFrame, n=10, path=None, verbose=True, title=None):
    assert n <= report.shape[0], "n <= {}".format(report.shape[0])
    ax = report.iloc[:n, 2:].plot.barh(stacked=True)
    ax.set_xlabel('Num')
    ax.set_ylabel('Client')
    ax.set_title(title)
    if path:
        plt.savefig(path, dpi=400, bbox_inches='tight')
    if verbose:
        plt.show()


def bubble_report(report: DataFrame, path=None, verbose=True):
    T = report.iloc[:, 2:]
    client_num, class_num = T.shape
    x = np.array(list(range(client_num)))
    y = np.array(list(range(class_num)))
    X, Y = np.meshgrid(x, y)
    Z = T.iloc[x, y]
    plt.scatter(X, Y, s=Z, c=X, cmap='viridis')
    plt.xlabel('Client')
    plt.ylabel('Class')
    plt.yticks(y)
    if path:
        plt.savefig(path, dpi=400, bbox_inches='tight')
    if verbose:
        plt.show()


# 客户端样本数统计
def client_sample_count(report: DataFrame, path=None, verbose=True):
    data = report.iloc[:, :2]
    sns.histplot(
        data=data,
        x="sample_num",
        edgecolor='none',
        alpha=0.7,
        shrink=0.95,
        color='#4169E1'
    )
    if path:
        plt.savefig(path, dpi=400, bbox_inches='tight')
    if verbose:
        plt.show()
