import torch
from statistics import median
from torch.utils.data import ConcatDataset
from config.env import SEED
from trainer.algo.robust.cmfl import CMFL, Role
from trainer.util.stats import gompertz
from utils.nn.aggregate import average
from utils.nn.functional import flatten, add, linear_sum, scalar_mul
from utils.select import random_select

def drop_one_average(states: list[dict]):
    assert len(states) > 1
    sum_state = linear_sum(states)
    for s in states:
        new_state = sub(sum_state, s)
        yield scalar_mul_(new_state, 1. / (len(states) - 1))

def val_score(g_acs, l_acs):
    g_acs, l_acs = set(g_acs), set(l_acs)
    return len(g_acs.intersection(l_acs)) / (len(g_acs.union(l_acs)) * 1.)


class CBRFL(CMFL):

    def _parse_kwargs(self, **kwargs):
        super(CBRFL, self)._parse_kwargs(**kwargs)
        if cbrfl := kwargs.get('cbrfl'):
            self.ac_thr = cbrfl.get('ac_thr', 0.)
            self.alpha = cbrfl.get('alpha', 0.)
            self.beta = cbrfl.get('beta', 0.)
            self.gamma = cbrfl.get('gamma', 1)

    def _init(self):
        super(CBRFL, self)._init()
        for cid in self._clients:
            self._clients[cid].scores = []
            self._clients[cid].tc = 1.0
            self._clients[cid].vc = 1.0
        self._ces_acs = dict()
        self._leader = None
        self._momentum = None

    def _local_update(self, cids):
        super(CMFL, self)._local_update(cids)

    def _consensus(self, tcs):
        hn = sum(1 for c in self._ces if self._clients[c].honest)
        mn = len(self._ces) - hn
        if hn == mn:
            self._leader = None
            return {}
        elif hn > mn:
            self._leader = random_select(
                [leader for leader in self._ces if self._clients[leader].honest]
                , 1, SEED + self._k)[0]
        else:
            self._leader = random_select(
                [leader for leader in self._ces if not self._clients[leader].honest]
                , 1, SEED + self._k)[0]
        return {
            t for t in tcs if sum(
                1 for c in self._ces_acs if t in self._ces_acs[c]
            ) > len(self._ces_acs) // 2
        }

    def _compute_loss(self, inputs: dict) -> dict:
        outputs = {}
        for idx, res in zip(inputs.keys(), self._pool.map(lambda a, v: a.evaluate.remote(*v), [
            (s, ConcatDataset([self._fds.val(cid), self._share_ds]), self.batch_size * 2 ** 5)
            for cid, s in inputs.values()
        ])):
            outputs[idx] = res[1]
        return outputs

    def _aggregate(self, cids):
        # 计算训练节点得分
        self._compute_score(cids)
        self._propose_acs(cids)
        # 共识结果
        ags = self._consensus(cids)
        if ags is None or len(ags) == 0:
            self._elect_committee(cids, set())
            return
        # 聚合更新全局模型
        for c in cids:
            self._clients[c].w = 0.
            if c in ags:
                self._clients[c].w = gompertz(median(self._clients[c].scores)).item()
                self._aggregator.update(self._clients[c].grad, self._clients[c].w)
        self._logger.info(f'[{self._k}] Aggregate: {self._fci(ags)}')
        grad = self._aggregator.compute()
        self._aggregator.reset()
        if self._momentum is None:
            self._momentum = grad
        V = linear_sum([self._momentum, grad], [self.alpha, 1. - self.alpha])
        # 调整学习率
        self.glr += torch.dot(flatten(self._momentum), flatten(V)).item()
        self._logger.info(f'[{self._k}] LR : {self.glr}')
        self._writer.add_scalar('metric/lr', self.glr, self._k)
        self._model.load_state_dict(
            add(scalar_mul(V, self.glr), self._state())
        )
        self._momentum = V
        # 计算训练贡献度
        self._compute_contribution(cids, ags, self.glr)
        # 选举新一届委员会
        self._elect_committee(cids, ags)

    def _propose_acs(self, cids):
        for i, c in enumerate(self._ces):
            ts = {t: self._clients[t].scores[i] for t in cids}
            self._ces_acs[c] = set([t for t, v in ts.items() if v >= self.ac_thr])
            self._logger.info(f'[{self._k}] Committee {self._fci([c])}: {self._ces_acs[c]}')

    def _elect_committee(self, tcs, acs):
        # 选举委员会
        first_cms = list([x for x in acs if self._clients[x].tc > 0.])
        first_ces = random_select(
            first_cms,
            min(len(self._ces) // 2, len(first_cms)),
            SEED + self._k,
            p=[self._clients[c].tc for c in first_cms]
        ) if len(first_cms) > 0 else []
        idlers = [c for c, v in self._clients.items() if v.role == Role.IDLER]
        second_cms = sorted(
            idlers, key=lambda x: self._clients[x].vc, reverse=True
        )[:(int(len(self._ces) * self.gamma)) - len(first_ces)]
        second_cms += [c for c in first_cms if c not in first_ces]
        second_ces = random_select(
            second_cms,
            len(self._ces) - len(first_ces),
            SEED + self._k,
            [self._clients[c].vc for c in second_cms]
        )
        cms = first_ces + second_ces
        for c in self._ces + tcs:
            if self._clients[c].role == Role.TRAINER:
                self._clients[c].scores.clear()
            self._clients[c].role = Role.IDLER
        self._ces_acs.clear()
        self._ces.clear()
        for c in cms:
            self._clients[c].role = Role.COMMITTEE
            self._ces.append(c)

    def _compute_score(self, tcs):
        grads = [self._clients[c].grad for c in tcs]
        mean_grad = average(grads)
        mean_state = add(mean_grad, self._state())
        mean_norm = torch.norm(flatten(mean_grad))
        c_losses = self._compute_loss({
            c: (c, mean_state) for c in self._ces
        })
        # 计算训练节点得分
        for c in self._ces:
            c_loss = c_losses[c]
            t_losses = self._compute_loss({
                t: (c, add(g, self._state())) for t, g in zip(tcs, drop_one_average(grads))
            })
            for t in tcs:
                t_loss = t_losses[t]
                t_norm = torch.norm(flatten(self._clients[t].grad))
                scale = mean_norm / t_norm / c_loss
                if t_norm > 0:
                    if not self._clients[c].honest and self.attack_kind == 'bg':
                        self._clients[t].scores.append(
                            (c_loss - t_loss) * scale
                        )
                    else:
                        self._clients[t].scores.append(
                            (t_loss - c_loss) * scale
                        )
                else:
                    self._clients[t].scores.append(0.)
        for t in tcs:
            scores = ', '.join(f'{s:.3f}' for s in self._clients[t].scores)
            self._logger.info(f'[{self._k}] Scores of {t}: [{scores}]')

    def _compute_contribution(self, tcs, acs, a=1.):
        for t in tcs:
            ptv = self._clients[t].w * a
            self._clients[t].tc = self.beta * ptv + (1. - self.beta) * self._clients[t].tc
            self._logger.info(f'[{self._k}] TC of {self._fci([t])}: {ptv:.4f}   {self._clients[t].tc:.4f}')
        for c in self._ces:
            jvc = val_score(acs, self._ces_acs[c]) * a
            if c == self._leader:
                self._clients[c].vc = self.beta * 2 * jvc + (1. - self.beta) * self._clients[c].vc
            else:
                self._clients[c].vc = self.beta * jvc + (1. - self.beta) * self._clients[c].vc
            self._logger.info(f'[{self._k}] VC of {self._fci([c])}: {jvc:.4f}   {self._clients[c].vc:.4f}')

    def _clean(self):
        self._ces.clear()
        self._ces_acs.clear()
        for cid in self._clients:
            del self._clients[cid].scores
        super(CBRFL, self)._clean()
