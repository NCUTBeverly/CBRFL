from abc import abstractmethod, ABC
from collections import OrderedDict

from torch.nn import Module
from utils.nn.aggregate import average
from utils.nn.functional import linear_sum, scalar_mul_


class StatelessAggregator(ABC):

    @abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def compute(self, *args, **kwargs) -> OrderedDict:
        raise NotImplementedError

    @abstractmethod
    def reset(self, *args, **kwargs):
        raise NotImplementedError


class StatefulAggregator(ABC):

    def __init__(self, model: Module):
        self._model = model
        self._model.eval()

    def get_state(self, copy=False) -> OrderedDict:
        state = OrderedDict()
        for k, v in self._model.state_dict().items():
            state[k] = v.clone() if copy else v
        return state

    def set_state(self, state: OrderedDict):
        self._model.load_state_dict(state)

    @abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def step(self, *args, **kwargs) -> OrderedDict:
        raise NotImplementedError

    @abstractmethod
    def reset(self, *args, **kwargs):
        raise NotImplementedError


class MeanAggregator(StatelessAggregator):

    def __init__(self):
        super(MeanAggregator, self).__init__()
        self._states, self._weights = [], []

    def update(self, state: OrderedDict, weight):
        self._states.append(state)
        self._weights.append(weight)

    def compute(self) -> OrderedDict:
        assert len(self._states) == len(self._weights) > 0
        return average(self._states, self._weights)

    def reset(self):
        self._weights.clear()
        self._states.clear()


class OnlineAggregator(StatelessAggregator):

    def __init__(self, size: int = 10):
        super(OnlineAggregator, self).__init__()
        self._state = None
        self._size = size
        self._weight = 0.
        self._k = -1

    def update(self, state: OrderedDict, weight: float):
        self._k = (self._k + 1) % self._size
        if self._state is None:
            self._state, self._weight = state, weight
        else:
            self._state = scalar_mul_(
                linear_sum([self._state, state], [self._weight, weight]),
                1. / (self._weight + weight)
            )
            self._weight += weight

    def compute(self) -> OrderedDict:
        assert self._k == self._size - 1
        return self._state

    def reset(self):
        self._state, self._weight = None, 0.
