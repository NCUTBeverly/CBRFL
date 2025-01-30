from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class Sent140(Dataset):

    def __init__(self, root, is_train=True, transform=None, target_transform=None):
        self.root = f"{root}/trainingandtestdata/"
        self.transform = transform
        self.target_transform = target_transform
        self._data, self._target = self._load(is_train)

    def _load(self, is_train=True):
        root = Path(self.root).joinpath(
            'training.1600000.processed.noemoticon.csv' if is_train else 'testdata.manual.2009.06.14.csv'
        )
        df = pd.read_csv(
            str(root),
            encoding='ISO-8859-1',
            names=['target', 'id', 'date', 'query', 'username', 'content']
        )
        return np.array(df.pop('content')), np.array(df.pop('target'))

    def __getitem__(self, index) -> T_co:
        data = self.transform(self._data[index]) if self.transform else self._data[index]
        target = self.target_transform(self._target[index]) if self.target_transform else self._target[index]
        return data, target

    def __len__(self):
        return self._target.shape[0]
