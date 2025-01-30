from collections import OrderedDict

import torch

from trainer.algo.robust.krum import Krum
from utils.nn.functional import flatten, add


def trimmed_mean(states: list[dict], num_attackers_selected: int = 2):
    """Aggregate weight updates from the clients using trimmed-mean."""
    flattened_states = torch.stack([flatten(s) for s in states])
    n, d = flattened_states.shape
    median_states = torch.median(flattened_states, dim=0)[0]
    sort_idx = torch.argsort(torch.abs(flattened_states - median_states), dim=0)
    sorted_states = flattened_states[sort_idx, torch.arange(d)[None, :]]
    mean_states = torch.mean(sorted_states[: n - 2 * num_attackers_selected], dim=0)
    start_index = 0
    trimmed_mean_update = OrderedDict()
    for name, weight_value in states[0].items():
        trimmed_mean_update[name] = mean_states[
                                    start_index: start_index + len(weight_value.view(-1))
                                    ].reshape(weight_value.shape)
        start_index = start_index + len(weight_value.view(-1))
    return trimmed_mean_update


class TrimmedMean(Krum):

    def _aggregate(self, cids):
        grads = [self._clients[cid].grad for cid in cids]
        krum_grad = trimmed_mean(grads, self.K)
        new_state = add(self._state(), krum_grad)
        self._model.load_state_dict(new_state)
