from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class DatasetWrapper(Dataset):

    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> T_co:
        data, target = self.dataset[index]
        data = self.transform(data) if self.transform else data
        target = self.target_transform(target) if self.target_transform else target
        return data, target


class FlipLabel:
    def __init__(self, offset, num_classes=10):
        self._offset = offset
        self._num_classes = num_classes

    def __call__(self, x):
        return (x - self._offset) % self._num_classes
