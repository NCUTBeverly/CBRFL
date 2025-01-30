from collections import OrderedDict
from torch.nn import init


def with_kaiming_normal(state: dict):
    new_state = OrderedDict()
    for ln in state:
        sc = state[ln].clone()
        if ln.endswith('.weight'):
            new_state[ln] = init.kaiming_normal_(sc)
        elif ln.endswith('.bias'):
            new_state[ln] = init.constant_(sc, 0.)
        else:
            new_state[ln] = sc
    return new_state


# weight
def with_sparse(state: dict, sparsity: float):
    new_state = OrderedDict()
    for ln in state:
        sc = state[ln].clone()
        if ln.endswith('.weight'):
            new_state[ln] = init.sparse_(sc, sparsity=sparsity)
        elif ln.endswith('.bias'):
            new_state[ln] = init.constant_(sc, 0.)
        else:
            new_state[ln] = sc
    return new_state
