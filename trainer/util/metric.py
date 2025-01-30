from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class Metric:
    num: int
    loss: float
    acc: float

    def __str__(self):
        return 'Loss: {:.3f}  Acc: {:.1%}'.format(self.loss, self.acc)


def average(metrics: Iterable[Metric]):
    nums = list(map(lambda v: v.num, metrics))
    if np.sum(nums) == 0:
        return Metric(0, 0., 0.)
    return Metric(
        np.sum(nums).item(),
        np.average(list(map(lambda v: v.loss, metrics)), weights=nums).item(),
        np.average(list(map(lambda v: v.acc, metrics)), weights=nums).item()
    )


class MetricAverager:

    def __init__(self):
        self._res = None
        self._metrics = []

    def update(self, m: Metric):
        if self._res:
            raise RuntimeError(f'Please reset {self.__class__.__name__} !')
        self._metrics.append(m)

    def compute(self):
        if self._res is None:
            self._res = average(self._metrics)
        return self._res

    def reset(self):
        self._metrics.clear()
        self._res = None
