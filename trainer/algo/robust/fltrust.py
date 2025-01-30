from torch import norm

from config.env import SEED
from trainer.algo.robust.base import RobustFL
from trainer.algo.robust.cmfl import CMFL, Role
from trainer.util.format import float_seq2str
from trainer.util.stats import relu
from utils.nn.aggregate import average
from utils.nn.functional import flatten, scalar_mul_, add
from utils.nn.stats import cosine_similarity
from utils.select import random_select


class FLTrust(CMFL):

    def _local_update(self, cids):
        super(CMFL, self)._local_update(cids)

    def _weights(self, tcs):
        self._trust_grad = self._trust_update()
        weights = [
            relu(cosine_similarity(flatten(self._trust_grad), flatten(self._clients[t].grad))).item()
            for t in tcs
        ]
        self._logger.info(f'Weight: {float_seq2str(weights)}')
        # self._normalize(tcs)
        return weights

    def _trust_update(self):
        self._pool.submit(lambda a, v: a.fit.remote(*v), (
            self._state(), self._share_ds, self.local_args
        ))
        return self._pool.get_next()[0]

    def _normalize(self, tcs):
        grads = [
            self._clients[cid].grad
            for cid in tcs if norm(
                flatten(self._clients[cid].grad)
            ).item() > 0.
        ]
        factors = [
            (norm(flatten(self._trust_grad)) / norm(flatten(g))).item()
            for g in grads
        ]
        self._logger.info(f'Factor: {float_seq2str(factors)}')
        for g, c in zip(grads, factors):
            scalar_mul_(g, c)

    def _aggregate(self, cids):
        weights = self._weights(cids)
        grads = [self._clients[cid].grad for cid in cids]
        ags = [cid for cid, w in zip(cids, weights) if w > 0.]
        self._logger.info(f'[{self._k}] Aggregate: {self._fci(ags)}')
        grad = average(grads, weights)
        self._model.load_state_dict(add(self._state(), grad))
        self._elect_committee(cids, ags)

    def _elect_committee(self, tcs, acs):
        selected = random_select(
            [k for k, v in self._clients.items() if v.role == Role.IDLER],
            len(self._ces),
            seed=SEED + self._k
        )
        for cid in tcs + self._ces:
            self._clients[cid].role = Role.IDLER
        self._ces.clear()
        for cid in selected:
            self._clients[cid].role = Role.COMMITTEE
            self._ces.append(cid)

