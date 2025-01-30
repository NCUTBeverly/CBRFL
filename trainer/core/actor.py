from abc import abstractmethod
from collections import OrderedDict

import ray
import torch
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from torchmetrics import SumMetric, MeanMetric, Accuracy
from benchmark.builder import build_optimizer
from config.env import PREFETCH_FACTOR, WORKER_NUM
from utils.nn.functional import sub


def dataloader(
        dataset: Dataset, batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False
):
    for data, target in DataLoader(
            dataset, batch_size,
            shuffle=shuffle,
            prefetch_factor=PREFETCH_FACTOR,
            num_workers=WORKER_NUM,
            drop_last=drop_last
    ):
        yield data, target


class CPUActor:

    def __init__(self, model: Module, criterion: Module):
        self.model = model
        self.criterion = criterion

    def get_state(self, copy=False):
        state = OrderedDict()
        for k, v in self.model.state_dict().items():
            state[k] = v.clone() if copy else v
        return state

    def set_state(self, state: OrderedDict):
        self.model.load_state_dict(state)

    def _setup_fit(self, args: dict):
        self._batch_size = args.get('batch_size', 32)
        self._epoch = args.get('epoch', 5)
        self._max_grad_norm = args.get('max_grad_norm', 10.0)
        self._opt = build_optimizer(
            args['opt'].get('name', 'SGD'),
            self.model.parameters(),
            args['opt'].get('args', {'lr': 0.001})
        )

    @torch.no_grad()
    def evaluate(self, state: OrderedDict, dataset: Dataset, batch_size: int):
        if dataset is None or len(dataset) == 0:
            return 0, 0., 0.
        num, loss, acc = SumMetric(), MeanMetric(), Accuracy()
        self.set_state(state)
        self.model.eval()
        for data, target in dataloader(dataset, batch_size):
            logit = self.model(data)
            num.update(target.shape[0])
            loss.update(self.criterion(logit, target))
            acc.update(logit, target)
        return num.compute().item(), loss.compute().item(), acc.compute().item()

    @abstractmethod
    def fit(self, state: OrderedDict, dataset: Dataset, args: dict):
        raise NotImplementedError


@ray.remote
class BasicActor(CPUActor):

    def fit(self, state: OrderedDict, dataset: Dataset, args: dict):
        self.set_state(state)
        self._setup_fit(args)
        self.model.train()
        for k in range(self._epoch):
            for data, target in dataloader(dataset, self._batch_size):
                self._opt.zero_grad()
                self.criterion(self.model(data), target).backward()
                clip_grad_norm_(self.model.parameters(), self._max_grad_norm)
                self._opt.step()
        return sub(self.get_state(), state), self.evaluate(self.get_state(), dataset, self._batch_size)


