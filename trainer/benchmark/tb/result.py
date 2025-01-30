from pathlib import Path

import numpy as np
from pandas import Series, DataFrame
from benchmark.tb.data import parse_tb_scalar
from utils.format import print_df

def topk_mean(arr, k=10):
    partition_idx = -k
    top_k = np.partition(arr, partition_idx)[partition_idx:]
    # 计算平均值
    return np.mean(top_k)

def get_acc_table(src: str, axis=0, verbose=False):
    tmp, data = {}, {}
    for d in filter(lambda x: x.is_dir() and not x.stem.startswith('.'), Path(src).iterdir()):
        print(f'Parsing {d.name}...')
        for t in filter(lambda x: not x.stem.startswith('.'), d.iterdir()):
            print(t.name)
            tmp[t.name] = [
                str(round(topk_mean(parse_tb_scalar(t.absolute(), 'test/{}'.format('acc'))[1]) * 100, 3))
            ]
            tmp[t.name].extend([
                str(round(topk_mean(parse_tb_scalar(st.absolute(), 'test/{}'.format('acc'))[1]) * 100, 3))
                for st in filter(lambda x: x.is_dir(), t.iterdir())
            ])
        data[d.name] = Series(data={k: '\t'.join(tmp[k]) for k in tmp})
        tmp.clear()
    df = DataFrame(data=data)
    df = DataFrame(df.values.T, index=df.columns, columns=df.index) if axis == 1 else df
    if verbose:
        print_df(df)
    return df


def get_loss_table(src: str, axis=0, verbose=False):
    tmp, data = {}, {}
    for d in filter(lambda x: x.is_dir() and not x.stem.startswith('.'), Path(src).iterdir()):
        for t in filter(lambda x: not x.stem.startswith('.'), d.iterdir()):
            tmp[t.name] = [
                str(round(min(parse_tb_scalar(t.absolute(), 'test/{}'.format('loss'))[1]), 3))
            ]
            tmp[t.name].extend([
                str(round(min(parse_tb_scalar(st.absolute(), 'test/{}'.format('loss'))[1]), 3))
                for st in filter(lambda x: x.is_dir(), t.iterdir())
            ])
        data[d.name] = Series(data={k: '\t'.join(tmp[k]) for k in tmp})
        tmp.clear()
    df = DataFrame(data=data)
    df = DataFrame(df.values.T, index=df.columns, columns=df.index) if axis == 1 else df
    if verbose:
        print_df(df)
    return df
