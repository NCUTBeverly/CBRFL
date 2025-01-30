import math
import numpy as np
import torch


class ToEMnistTarget:
    def __init__(self, split):
        self.__split = split

    def __call__(self, x):
        if self.__split == 'letters':
            return x - 1
        else:
            return x


class ToNumpy:
    def __init__(self, dtype):
        self.__dtype = dtype

    def __call__(self, x):
        return np.array(x, dtype=self.__dtype)


class ToVector:
    LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"

    def __init__(self, mode=True):
        self.mode = mode

    def __call__(self, x):
        if self.mode:
            return torch.tensor(list(map(lambda ch: self.LETTERS.find(ch), x)))
        else:
            return torch.tensor(self.LETTERS.find(x))


class ToSent140Content:

    def __init__(self, index=-1):
        self._index = index

    def __call__(self, x):
        return x[self._index]


class ToSent140Target:

    def __call__(self, x):
        return int(math.log2(max(int(x), 1)))


class ToCelebaAttrTarget:
    def __init__(self, col=0):
        assert col < 40
        self._col = col

    def __call__(self, x):
        return x[self._col]


class RemapLabel:

    def __init__(self, num_classes=10):
        self.num_classes = num_classes

    def __call__(self, x):
        return x % self.num_classes


class FlipLabel:

    def __init__(self, p=0.5, num_classes=10, seed=2077):
        self.p = p
        self.num_classes = num_classes
        np.random.seed(seed)

    def __call__(self, x):
        if np.random.rand(1) < self.p:
            return (x - 3) % self.num_classes
        return x


class ToFemnist:

    def __call__(self, x):
        return np.array(x, np.float32).reshape((-1, 28, 28))
