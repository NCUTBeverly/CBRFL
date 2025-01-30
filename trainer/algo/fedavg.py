from ray.util import ActorPool
from torch.nn import CrossEntropyLoss
from config.env import ACTOR_NUM, SEED
from trainer.core.actor import BasicActor
from trainer.core.aggregator import MeanAggregator
from trainer.core.trainer import FLTrainer
from trainer.util.metric import MetricAverager, Metric
from utils.nn.functional import add_, scalar_mul_
from utils.select import random_select
from utils.nn.init import with_kaiming_normal


class FedAvg(FLTrainer):

    def _state(self, cid=None):
        return self._model.state_dict()

    def _parse_kwargs(self, **kwargs):
        super(FedAvg, self)._parse_kwargs(**kwargs)
        self.sample_rate = kwargs.get('sample_rate', 0.1)
        self.glr = kwargs.get('glr', 1.)
        self.epoch = kwargs.get('epoch', 5)
        self.batch_size = kwargs.get('batch_size', 32)
        self.max_grad_norm = kwargs.get('max_grad_norm', 10.0)
        self.opt = kwargs.get('opt', {'lr': 0.002})
        self.local_args = {
            'opt': self.opt,
            'batch_size': self.batch_size,
            'epoch': self.epoch,
            'max_grad_norm': self.max_grad_norm
        }

    def _init(self):
        super(FedAvg, self)._init()
        self._model.load_state_dict(with_kaiming_normal(self._model.state_dict()))
        self._aggregator = self._build_aggregator()
        self._pool = self._build_actor_pool()
        self._metric_averager = MetricAverager()

    def _select_client(self):
        selected = random_select(self._clients, self.sample_rate, self._k + SEED)
        self._logger.info(f'[{self._k}] Selected: {selected}')
        return selected

    def _local_update(self, cids):
        self._metric_averager.reset()
        for cid, res in zip(cids, self._pool.map(lambda a, v: a.fit.remote(*v), self._local_update_args(cids))):
            self._local_update_hook(cid, res)
            self._metric_averager.update(Metric(*res[1]))
        self._handle_metric(self._metric_averager.compute(), 'train', self._writer)

    def _aggregate(self, cids):
        grad = self._aggregator.compute()
        self._aggregator.reset()
        state = add_(scalar_mul_(grad, self.glr), self._state())
        self._model.load_state_dict(state)

    def _val(self, cids):
        self._metric_averager.reset()
        for cid, res in zip(cids, self._pool.map(lambda a, v: a.evaluate.remote(*v), [
            (self._state(cid), self._fds.val(cid), self.batch_size) for cid in cids
        ])):
            self._metric_averager.update(Metric(*res))
        self._handle_metric(self._metric_averager.compute(), 'val', self._writer)

    def _test(self):
        self._metric_averager.reset()
        self._pool.submit(lambda a, v: a.evaluate.remote(*v), (
            self._state(), self._fds.test(), self.batch_size * ACTOR_NUM
        ))
        self._metric_averager.update(Metric(*self._pool.get_next()))
        self._handle_metric(self._metric_averager.compute(), 'test', self._writer)

    def _clean(self):
        self._writer.close()
        self._aggregator.reset()
        self._metric_averager.reset()
        super(FedAvg, self)._clean()

    def _build_actor_pool(self):
        return ActorPool([
            BasicActor.remote(self._model, CrossEntropyLoss())
            for _ in range(ACTOR_NUM)
        ])

    def _build_aggregator(self):
        return MeanAggregator()

    def _local_update_args(self, cids):
        return [(self._state(c), self._fds.train(c), self.local_args) for c in cids]

    def _local_update_hook(self, cid, res):
        self._aggregator.update(res[0], res[1][0])
