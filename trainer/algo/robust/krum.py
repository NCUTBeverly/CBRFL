import torch
from trainer.algo.robust.base import RobustFL
from utils.nn.functional import flatten, unflatten, extract_shape, add


def krum(states: list[dict], num_attackers_selected: int = 2):
    assert len(states) > num_attackers_selected
    remaining_states = torch.stack([flatten(s) for s in states])
    distances = []
    for s in remaining_states:
        distance = torch.norm((remaining_states - s), dim=1) ** 2
        distances = (
            distance[None, :]
            if not len(distances)
            else torch.cat((distances, distance[None, :]), 0)
        )
    distances = torch.sort(distances, dim=1)[0]
    scores = torch.sum(
        distances[:, : len(remaining_states) - 2 - num_attackers_selected], dim=1
    )
    sorted_scores = torch.argsort(scores)
    c_states = remaining_states[sorted_scores[0]][None, :]
    return unflatten(c_states.view(-1), extract_shape(states[0]))


class Krum(RobustFL):

    def _parse_kwargs(self, **kwargs):
        super(Krum, self)._parse_kwargs(**kwargs)
        if kr := kwargs.get("krum", {}):
            self.K = kr.get("K", 2)

    def _aggregate(self, cids):
        grads = [self._clients[cid].grad for cid in cids]
        krum_grad = krum(grads, self.K)
        new_state = add(self._state(), krum_grad)
        self._model.load_state_dict(new_state)

    def _clean(self):
        super(RobustFL, self)._clean()
