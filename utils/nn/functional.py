from collections import OrderedDict
from functools import reduce
from operator import mul
from typing import Optional, Sequence

import torch
from torch import Tensor


@torch.no_grad()
def numel(state: dict):
    return sum(map(lambda x: x.numel(), state.values()))


@torch.no_grad()
def extract_layer(state: dict, *layer_names):
    layer_names = [ln for ln in layer_names if ln in state]
    sub_state = OrderedDict()
    for ln in layer_names:
        sub_state[ln] = state[ln]
    return sub_state


@torch.no_grad()
def extract_shape(state: dict):
    dim = OrderedDict()
    for ln in state:
        dim[ln] = tuple(state[ln].shape)
    return dim


@torch.no_grad()
def state2vector(states: Sequence[dict]):
    with torch.no_grad():
        vector = list(map(lambda x: flatten(x), states))
    return vector


@torch.no_grad()
def flatten(state: dict):
    return torch.cat(list(map(lambda x: x.flatten(), state.values())))


@torch.no_grad()
def unflatten(vector: Tensor, shape: dict):
    layer_size = [reduce(mul, v, 1) for v in shape.values()]
    new_state = OrderedDict()
    for ln, l in zip(shape, torch.split(vector, layer_size)):
        new_state[ln] = torch.reshape(l, shape[ln])
    return new_state


@torch.no_grad()
def zero_like(state):
    return sub(state, state)


@torch.no_grad()
def equal(state1: dict, state2: dict):
    for ln in state1:
        if not torch.equal(state1[ln], state2[ln]):
            return False
    return True


@torch.no_grad()
def add(state1: dict, state2: dict, alpha=1.):
    new_state = OrderedDict()
    for ln in state1:
        new_state[ln] = state1[ln] + state2[ln] * alpha
    return new_state


@torch.no_grad()
def add_(state1: dict, state2: dict, alpha=1.):
    for ln in state1:
        state1[ln] += state2[ln] * alpha
    return state1


@torch.no_grad()
def sub(state1: dict, state2: dict):
    new_state = OrderedDict()
    for ln in state1:
        new_state[ln] = state1[ln] - state2[ln]
    return new_state


@torch.no_grad()
def sub_(state1: dict, state2: dict):
    for ln in state1:
        state1[ln] -= state2[ln]
    return state1


@torch.no_grad()
def scalar_mul(state: dict, scalar):
    new_state = OrderedDict()
    for ln in state:
        new_state[ln] = state[ln] * scalar
    return new_state


@torch.no_grad()
def scalar_mul_(state: dict, scalar):
    for ln in state:
        state[ln] *= scalar
    return state


@torch.no_grad()
def linear_sum(states: Sequence[dict], weights: Optional[Sequence] = None):
    assert len(states) > 0
    new_state = OrderedDict()
    if weights is None:
        weights = torch.ones(len(states))
    for ln in states[0]:
        new_state[ln] = reduce(
            torch.add,
            map(lambda x: x[0][ln] * x[1], zip(states, weights))
        )
    return new_state


@torch.no_grad()
def proj_(state1: dict, state2: dict):
    fs1, fs2 = flatten(state1), flatten(state2)
    proj_direct = torch.dot(fs1, fs2) / torch.norm(fs2) ** 2
    fs1 -= min(proj_direct, 0.) * fs2
    return unflatten(fs1, extract_shape(state1))


@torch.no_grad()
def proj(state1: dict, state2: dict):
    fs1, fs2 = flatten(state1), flatten(state2)
    pj12 = torch.dot(fs1, fs2) / torch.dot(fs2, fs2) * fs2
    return unflatten(pj12, extract_shape(state1))


@torch.no_grad()
def powerball(state: dict, gamma: float):
    new_state = OrderedDict()
    for ln in state:
        new_state[ln] = torch.sign(state[ln]) * torch.pow(torch.abs(state[ln]), gamma)
    return new_state
