from collections.abc import Sequence
import numpy as np
from torch import Tensor


def _np_array(data):
    if isinstance(data, Sequence):
        if isinstance(data[0], Tensor):
            return np.array([d.numpy() for d in data])
        elif isinstance(data[0], (float, int)):
            return np.array(list(data))
    elif isinstance(data, Tensor):
        return data.numpy()
    elif isinstance(data, (float, int)):
        return np.array([data])
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise TypeError(f'Unsupported type {type(data)}')


def gompertz(x, a=1, b=-4, c=-6):
    x = _np_array(x)
    return a * np.exp(b * np.exp(c * x))

def relu(x):
    x = _np_array(x)
    return np.maximum(x, 0)

def exp_decay(x, a=1, b=0.1):
    x = _np_array(x)
    return a * np.exp(-b * x)


def dtanh(x, a=1., b=1):
    x = _np_array(x)
    return a - np.tanh(b * x) ** 2


def relu_(x, a=1, b=1):
    x = _np_array(x)
    return (np.exp(-a * x) - 1) * (x < 0.) + b


def ratio(x):
    x = _np_array(x)
    min_x = np.min(x)
    sum_x = np.sum(x - min_x)
    return (x - min_x) / sum_x
