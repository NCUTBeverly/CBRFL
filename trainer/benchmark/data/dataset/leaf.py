from typing import Sequence
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataset import T_co
from torchvision.transforms import Compose

from utils.io import load_jsons
from utils.data.fed import FederatedDataset
from benchmark.data.dataset.transformer import ToNumpy, ToVector, ToFemnist, ToSent140Target, ToSent140Content
from utils.nlp import ToFixedSeq


class SequenceDataset(Dataset):

    def __init__(self, datasource: Sequence, transform=None, target_transform=None):
        self.datasource = datasource
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        if isinstance(self.datasource, dict):
            return len(self.datasource['y'])
        return len(self.datasource)

    def __getitem__(self, index) -> T_co:
        if isinstance(self.datasource, dict):
            data, target = self.datasource['x'][index], self.datasource['y'][index]
        else:
            data, target = self.datasource[index]
        data = self.transform(data) if self.transform else data
        target = self.target_transform(target) if self.target_transform else target

        return data, target


class LEAF(FederatedDataset):

    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self._train_data = self._load('train')
        self._test_data = self._load('test')
        self._users = self._train_data['users']

    def __len__(self):
        return len(self._users)

    def __iter__(self):
        for user in self._users:
            yield user

    def __contains__(self, key):
        return key in self._users

    def _load(self, tag='train'):
        data = {}
        for js in load_jsons(f"{self.root}/{tag}/"):
            data.update(js)
        return data

    def train(self, key) -> Dataset:
        return SequenceDataset(self._train_data['user_data'][key], self.transform, self.target_transform)

    def val(self, key) -> Dataset:
        return SequenceDataset(self._test_data['user_data'][key], self.transform, self.target_transform)

    def test(self) -> Dataset:
        return ConcatDataset([self.val(key) for key in self._users])


class Synthetic(LEAF):

    def __init__(self, root):
        super(Synthetic, self).__init__(root, ToNumpy(np.float32), ToNumpy(np.int64))


class Sent140(LEAF):

    def __init__(self, root, max_len=35, dim=25):
        transform = Compose([
            ToSent140Content(-2),
            ToFixedSeq(max_len=max_len, dim=dim)
        ])
        target_transform = ToSent140Target()
        super(Sent140, self).__init__(root, transform, target_transform)


class Shakespeare(LEAF):

    def __init__(self, root):
        super(Shakespeare, self).__init__(root, ToVector(), ToVector(False))


class Femnist(LEAF):

    def __init__(self, root):
        super(Femnist, self).__init__(root, ToFemnist())
