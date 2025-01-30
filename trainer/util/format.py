from collections.abc import Sequence
from typing import Union, Any
from torch import Tensor


def float_seq2str(f: Sequence[Union[float, Tensor]], precision: int = 3):
    return str([
        round(x.item() if isinstance(x, Tensor) else x, precision)
        for x in f
    ])


def float_dict2str(f: dict[Any, Union[float, Tensor]], precision: int = 3):
    return str({
        k: round(x.item() if isinstance(x, Tensor) else x, precision)
        for k, x in f.items()
    })


