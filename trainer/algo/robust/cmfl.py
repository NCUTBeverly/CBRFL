import math
import statistics
from enum import Enum

import torch
from torch import norm
from torch.nn import Module

from config.env import SEED
from trainer.algo.robust.base import RobustFL
from utils.data.fed import FederatedDataset
from utils.nn.functional import flatten, scalar_mul
from utils.select import random_select


class Role(Enum):
    """
    CMFL 节点身份枚举类型
    """
    IDLER = -1
    TRAINER = 0
    # 委员会成员
    COMMITTEE = 1
    LEADER = 2
    FOLLOWER = 3


class CMFL(RobustFL):

    def _init(self):
        super(CMFL, self)._init()

        for cid in self._clients:
            self._clients[cid].role = Role.IDLER
            self._clients[cid].grad = None
            self._clients[cid].w = 0
        self._ces = random_select(
            list(self._clients.keys()),
            self.ce_rate * self.sample_rate,
            seed=SEED + self._k
        )
        for cid in self._ces:
            self._clients[cid].role = Role.COMMITTEE
        self._hc_num = 0.

    def _parse_kwargs(self, **kwargs):
        super(CMFL, self)._parse_kwargs(**kwargs)
        if cmfl := kwargs['cmfl']:
            self.ce_rate = cmfl.get('ce_rate', 0.4)
            self.agg_rate = cmfl.get('agg_rate', 0.4)
            self.high = cmfl.get('high', True)

    def _select_client(self):
        # 随机选择节点数量
        select_num = int(len(self._clients) * self.sample_rate)
        selected = random_select(
            [k for k, v in self._clients.items() if v.role == Role.IDLER],
            select_num - len(self._ces),
            seed=SEED + self._k
        )
        for cid in selected:
            self._clients[cid].role = Role.TRAINER
        self._logger.info(f'[{self._k}] Trainer: {self._fci(selected)}')
        self._logger.info(f'[{self._k}] Committee: {self._fci(self._ces)}')

        self._hc_num += len([c for c in self._ces if self._clients[c].honest]) * 1.
        self._logger.info(f'[{self._k}] Host Num: {self._hc_num}')
        self._writer.add_scalar('metric/host_num', self._hc_num / ((self._k + 1) * len(self._ces)), self._k)
        return selected

    def _local_update(self, cids):
        super(CMFL, self)._local_update(cids)
        for cid, res in zip(self._ces, self._pool.map(
                lambda a, v: a.fit.remote(*v),
                self._local_update_args(self._ces)
        )):
            self._local_update_hook(cid, res)

    def _consensus(self, tcs):
        tk = int(len(tcs) * self.agg_rate)
        ordered_tcs = sorted(tcs, key=lambda i: self._clients[i].score, reverse=self.high)
        acs = ordered_tcs[:tk]
        return acs

    def _aggregate(self, cids):
        # 计算训练节点得分
        self._compute_score(cids)
        # 共识结果
        acs = self._consensus(cids)
        # 聚合更新全局模型
        super(CMFL, self)._aggregate(acs)
        # 选举新一届委员会
        self._elect_committee(tcs=cids, acs=acs)

    def _clean(self):
        self._ces.clear()
        for cid in self._clients:
            del self._clients[cid].role
            del self._clients[cid].w
            del self._clients[cid].grad
        super(CMFL, self)._clean()

    def _clip_median(self, tcs):
        m_norm = statistics.median([norm(flatten(self._clients[c].grad)).item() for c in tcs])
        for c in tcs:
            c_norm = norm(flatten(self._clients[c].grad))
            if c_norm > m_norm:
                self._clients[c].grad = scalar_mul(self._clients[c].grad, m_norm / c_norm)

    def _compute_score(self, tcs):
        ccs = self._ces
        # 计算训练节点得分
        for k in tcs:
            Pk = [
                torch.norm(flatten(self._clients[k].grad) - flatten(self._clients[c].grad), p=2).item()
                for c in ccs
            ]
            self._clients[k].score = len(ccs) / (sum(Pk) + 1e-8)
        self._logger.info(f'[{self._k}] Score: {[self._clients[k].score for k in tcs]}')

    def _elect_committee(self, tcs, acs):
        C = len(self._ces)
        ordered_tcs = sorted(tcs, key=lambda i: self._clients[i].score, reverse=self.high)
        # 选举新一界委员会
        for c in ordered_tcs[C:]:
            self._clients[c].role = Role.IDLER
        for c in self._ces:
            self._clients[c].role = Role.IDLER
        self._ces.clear()
        for c in ordered_tcs[:C]:
            self._clients[c].role = Role.COMMITTEE
            self._ces.append(c)
