import pickle
from collections import OrderedDict
import torch


class Param(OrderedDict):

    def __init__(self, state: dict):
        super().__init__()
        self.update(state)

    def __neg__(self):
        return -1 * self

    def __add__(self, other):
        ret = OrderedDict()
        for ln in self.keys():
            ret[ln] = self[ln] + other[ln]
        return Param(ret)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        ret = OrderedDict()
        for ln in self.keys():
            ret[ln] = self[ln] - other[ln]
        return Param(ret)

    def __eq__(self, other):
        if not all([t == o for t, o in zip(self.keys(), other.keys())]):
            return False
        for ln in self.keys():
            if torch.all(torch.ne(self[ln], other[ln])):
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        ret = OrderedDict()
        for ln in self.keys():
            ret[ln] = self[ln] * other
        return Param(ret)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        ret = OrderedDict()
        for ln in self.keys():
            ret[ln] = self[ln] * 1. / other
        return Param(ret)

    def __rdiv__(self, other):
        self.__truediv__(other)

    def __len__(self):
        return self.to_vector().shape[0]

    def __iter__(self):
        for ln in self.keys():
            yield ln, self[ln]

    def to_vector(self):
        return torch.cat(list(map(lambda x: x.flatten(), self.values())))

    def to_pkl(self, fn):
        pickle.dump(self, fn)

    def from_dict(self, state: dict):
        self.update(state)
        return self
