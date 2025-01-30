import numpy as np
from typing import Sequence


def random_select(candidates: Sequence, select, seed=None, p=None):
    np.random.seed(seed)
    if p is not None:
        p = np.array(p) / np.sum(p)
    total_num = len(candidates)
    select_num = select if isinstance(select, int) else int(total_num * select)
    selected_indices = np.random.choice(total_num, select_num, False, p=p)
    selected = [candidates[i] for i in selected_indices]
    return selected


def ribbon_select(candidates: Sequence, ratio=1., seed=None):
    np.random.seed(seed)
    total_num = len(candidates)
    select_num = int(total_num * ratio)
    if select_num >= total_num:
        return candidates
    step = total_num // select_num
    start_idx = np.random.randint(0, step)
    selected = [candidates[start_idx + i * step] for i in range(select_num)]
    return selected
