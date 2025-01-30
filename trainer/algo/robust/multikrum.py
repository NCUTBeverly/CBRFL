import torch

from trainer.algo.robust.base import RobustFL
from trainer.algo.robust.krum import Krum
from collections import OrderedDict

from utils.nn.functional import flatten, add


def multi_krum(states: list[dict], num_attackers_selected: int = 2):
    """Aggregate weight updates from the clients using multi-krum."""
    remaining_states = torch.stack([flatten(s) for s in states])
    candidates = []
    # Search for candidates based on distance
    while len(remaining_states) > 2 * num_attackers_selected + 2:
        distances = []
        for weight in remaining_states:
            distance = torch.norm((remaining_states - weight), dim=1) ** 2
            distances = (
                distance[None, :]
                if not len(distances)
                else torch.cat((distances, distance[None, :]), 0)
            )
        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(distances[:, : len(remaining_states) - 2 - num_attackers_selected], dim=1)
        indices = torch.argsort(scores)
        candidates = (
            remaining_states[indices[0]][None, :]
            if not len(candidates)
            else torch.cat(
                (candidates, remaining_states[indices[0]][None, :]), 0
            )
        )
        # Remove candidates from remaining
        remaining_states = torch.cat(
            (
                remaining_states[: indices[0]],
                remaining_states[indices[0] + 1:],
            ), 0
        )
    mean_state = torch.mean(candidates, dim=0)
    # Update global model
    start_index = 0
    mkrum_state = OrderedDict()
    for ln, w in states[0].items():
        mkrum_state[ln] = mean_state[start_index: start_index + len(w.view(-1))].reshape(w.shape)
        start_index = start_index + len(w.view(-1))
    return mkrum_state


class MultiKrum(Krum):

    def _aggregate(self, cids):
        grads = [self._clients[cid].grad for cid in cids]
        krum_grad = multi_krum(grads, self.K)
        new_state = add(self._state(), krum_grad)
        self._model.load_state_dict(new_state)
