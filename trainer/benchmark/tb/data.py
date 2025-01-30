import os.path
from glob import glob, iglob
from pathlib import Path
from pandas import DataFrame
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def parse_tb_scalar(path, scalar: str):
    steps, values = [], []
    ea = EventAccumulator(str(path), size_guidance={scalar: 0})
    ea.Reload()
    sks = ea.scalars.Keys()
    if scalar not in sks:
        raise KeyError(scalar)
    for it in ea.scalars.Items(scalar):
        steps.append(it.step)
        values.append(it.value)
    return steps, values


def get_scalar(src: str, scalar: str, step: int = 1, round=300, sub: bool = False, ignores: list[str] = None):
    if ignores is None:
        ignores = []
    data = dict()
    # 匹配非点文件夹
    paths = [Path(d) for d in filter(lambda x: os.path.isdir(x), glob(src))]
    paths = list(filter(lambda p: p.name not in ignores, paths))
    for d in paths:
        print(f'Parsing {d.name}...')
        tmp = parse_tb_scalar(d.absolute(), scalar)[1]
        if len(tmp) > round:
            tmp = tmp[:round]
        else:
            tmp += [None] * (round - len(tmp))
        data[d.name] = tmp
        for sd in filter(lambda x: x.is_dir() and sub, d.iterdir()):
            tmp = parse_tb_scalar(sd.absolute(), scalar)[1]
            delta_n = (len(data[d.name]) - len(tmp))
            data[f'{d.name}/{sd.name}'] = tmp[:delta_n] if delta_n < 0 else [None] * delta_n + tmp
    df = DataFrame(data)
    df.index *= step
    return df
