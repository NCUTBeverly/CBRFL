from collections import OrderedDict

import ray
import torch
from ray.util import ActorPool
from torch.nn import CrossEntropyLoss, Module
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset

from config.env import ACTOR_NUM
from trainer.algo.fedavg import FedAvg
from trainer.core.actor import CPUActor, dataloader
from utils.nn.functional import flatten, sub


@ray.remote
class ProxActor(CPUActor):

    def __init__(self, model: Module, criterion: Module, alpha: float):
        super().__init__(model, criterion)
        self._alpha = alpha

    def fit(self, state: OrderedDict, dataset: Dataset, args: dict):
        self.set_state(state)
        self._setup_fit(args)
        self.model.train()
        for k in range(self._epoch):
            for data, target in dataloader(dataset, self._batch_size):
                self._opt.zero_grad()
                loss = self.criterion(self.model(data), target) + self.__rt(state)
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self._max_grad_norm)
                self._opt.step()
        return sub(self.get_state(), state), self.evaluate(self.get_state(), dataset, self._batch_size)

    def __rt(self, global_state: OrderedDict):
        return .5 * self._alpha * torch.sum(torch.pow(flatten(self.get_state()) - flatten(global_state), 2))


class FedProx(FedAvg):

    def _parse_kwargs(self, **kwargs):
        super(FedProx, self)._parse_kwargs(**kwargs)
        if prox := kwargs['prox']:
            self.alpha = prox.get('alpha', 0.01)

    def _build_actor_pool(self):
        return ActorPool([
            ProxActor.remote(self._model, CrossEntropyLoss(), self.alpha)
            for _ in range(ACTOR_NUM)
        ])
