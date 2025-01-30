import os
import sys
import random
import shutil
import time
import traceback
from typing import Sequence
from importlib import import_module
import torch
import numpy as np


def set_seed(seed, has_torch=False, has_cuda=False):
    random.seed(seed)
    np.random.seed(seed)
    if has_torch:
        torch.manual_seed(seed)
        if has_cuda:
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def func_name():
    return sys._getframe().f_back.f_code.co_name


def force_exit(code):
    return os._exit(code)


def os_platform():
    return sys.platform


def delete_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        return True
    return False


def get_time():
    return time.strftime('%Y-%m-%d@%H:%M:%S', time.localtime(time.time()))


def locate(modules: Sequence[str], name: str, args: dict):
    for m in modules:
        try:
            return getattr(import_module(m), name)(**args)
        except:
            # debug
            print(traceback.format_exc())
            continue
    raise ImportError(f'The {name} is not found !!!')


def timestamp():
    return int(time.time())
