import pandas as pd
import torch
from torch.utils.data import random_split

from config.env import SEED, ACTOR_NUM
from trainer.algo.fedavg import FedAvg
from trainer.algo.robust.attack import DatasetWrapper, FlipLabel
from trainer.core.node import Client
from trainer.util.metric import Metric
from utils.nn.functional import scalar_mul
from utils.select import random_select


class RobustFL(FedAvg):

    def _parse_kwargs(self, **kwargs):
        super(RobustFL, self)._parse_kwargs(**kwargs)
        self.attack_kind = kwargs.get("attack_kind", "wo")
        self.attack_rate = kwargs.get("attack_rate", 0.0)
        self.sds_rate = kwargs.get("sds_rate", 0.0)

    def _fci(self, cids):
        if len(cids) == 1:
            cids = list(cids)
            return f"{cids[0]}({('H' if self._clients[cids[0]].honest else 'A')})"
        # 使用列表推导和条件表达式来构建字符串列表
        client_info_list = [
            f"{cid}({('H' if self._clients[cid].honest else 'A')})"
            for cid in cids
        ]
        return "[" + ", ".join(client_info_list) + "]"

    def _init(self):
        super(RobustFL, self)._init()
        self._clients = {cid: Client(id=cid, honest=True) for cid in self._fds}
        for cid in random_select(list(self._clients.keys()), self.attack_rate, seed=SEED - 1):
            self._clients[cid].honest = False
        self._share_ds, self._test_ds = random_split(
            self._fds.test(), [self.sds_rate, (1 - self.sds_rate)],
            torch.Generator().manual_seed(SEED)
        )

    def _local_update_hook(self, cid, res):
        self._clients[cid].w = res[1][0]
        if not self._clients[cid].honest and self.attack_kind == 'sv':
            self._clients[cid].grad = scalar_mul(res[0], 0.0)
        elif not self._clients[cid].honest and self.attack_kind == 'bg':
            self._clients[cid].grad = scalar_mul(res[0], -1.)
        else:
            self._clients[cid].grad = res[0]

    def _select_client(self):
        selected = random_select(list(self._clients.keys()), self.sample_rate, self._k + SEED)
        self._logger.info(f'[{self._k}] Selected: {self._fci(selected)}')
        return selected

    def _local_update_args(self, cids):
        local_update_args = []
        for cid in cids:
            if not self._clients[cid].honest and self.attack_kind == 'lf':
                ds = DatasetWrapper(self._fds.train(cid), target_transform=FlipLabel(3))
            else:
                ds = self._fds.train(cid)
            local_update_args.append((self._state(cid), ds, self.local_args))
        return local_update_args

    def _aggregate(self, cids):
        self._logger.info(f'[{self._k}] Aggregate: {self._fci(cids)}')
        for cid in cids:
            self._aggregator.update(self._clients[cid].grad, self._clients[cid].w)
        super(RobustFL, self)._aggregate(cids)

    def _test(self):
        self._metric_averager.reset()
        self._pool.submit(lambda a, v: a.evaluate.remote(*v), (
            self._state(), self._test_ds, self.batch_size * ACTOR_NUM
        ))
        self._metric_averager.update(Metric(*self._pool.get_next()))
        self._handle_metric(self._metric_averager.compute(), 'test', self._writer)

    def _clean(self):
        for cid in self._clients:
            del self._clients[cid].id
        df = pd.DataFrame(self._clients).T
        df.to_csv(f"{self._writer.log_dir}/tab.csv")
        super(RobustFL, self)._clean()
