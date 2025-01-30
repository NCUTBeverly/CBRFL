import random
from collections import defaultdict
from typing import Optional, Sequence

import torch
from utils.nn.functional import linear_sum, flatten, unflatten, extract_shape


@torch.no_grad()
def average(states: Sequence[dict], weights: Optional[Sequence] = None):
    if weights is None:
        weights = torch.ones(len(states))
    elif not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights)
    return linear_sum(states, weights / weights.sum())


@torch.no_grad()
def shuffle_layer(states: Sequence[dict], layers: Optional[Sequence[str]] = None, seed=None):
    assert len(states) > 0
    layers = layers if layers else states[0].keys()
    groups = defaultdict(list)
    random.seed(seed)
    for ln in layers:
        groups[ln.split('.')[0]].append(ln)
    for g in groups:
        indices = random.sample(range(len(states)), len(states))
        for i, s in enumerate(states):
            s_ = states[indices[i]]
            for ln in groups[g]:
                s[ln], s_[ln] = s_[ln], s[ln]
    return states


@torch.no_grad()
def pc_grad(states: Sequence[dict]):
    def _proj(state: dict):
        fs = flatten(state)
        for os in states:
            fos = flatten(os)
            proj_direct = torch.dot(fs, fos) / torch.norm(fos) ** 2
            fs -= min(proj_direct, 0.) * fos
        return unflatten(fs, extract_shape(state))

    if states is None or len(states):
        raise ValueError(states)
    return list(map(_proj, states))
