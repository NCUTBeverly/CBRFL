from typing import Optional
from torch.nn import Module

from utils.data.fed import FederatedDataset
from utils.tool import locate


def build_optimizer(name: str, params, args: dict):
    args = dict(params=params, **args)
    return locate([
        f'torch.optim',
        f'util.optim'
    ], name, args)


def build_model(name: str, args: Optional[dict] = None):
    args = dict(args) if args else dict()
    if len(name.split('.')) > 1:
        mods = name.split('.')
        name = mods[-1]
        module_path = '.'.join(mods[:-1])
    else:
        module_path = name.lower()
    return locate([
        f'benchmark.model.{module_path}',
    ], name, args)


def build_federated_dataset(name: str, args: Optional[dict] = None):
    args = dict(args) if args else dict()
    return locate([
        f'benchmark.data.fed.main_fds',
        f'benchmark.data.fed.inc_fds',
        f'benchmark.data.fed.cfl_fds',
    ], name, dict(args=args))


def build_trainer(name: str, net: Module, fds: FederatedDataset, args: dict):
    args = dict(model=net, fds=fds, **args)
    if len(name.split('.')) > 1:
        mods = name.split('.')
        name = mods[-1]
        mods[-1] = mods[-1].lower()
        module_path = '.'.join(mods)
    else:
        module_path = name.lower()
    return locate([
        f'trainer.algo.{module_path}'
    ], name, args)
