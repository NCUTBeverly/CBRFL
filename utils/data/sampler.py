import random
from typing import Sequence

from torch.utils.data import Sampler


class SubsetSampler(Sampler):

    def __init__(self, indices: Sequence[int], shuffle=False, seed=None):
        super(SubsetSampler, self).__init__(None)
        if not isinstance(indices, list):
            indices = list(indices)
        self._indices = indices
        if shuffle:
            random.shuffle(self._indices, seed)

    def __iter__(self):
        for ind in self._indices:
            yield ind

    def __len__(self):
        return len(self._indices)
