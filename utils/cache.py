import os
from cachetools import LRUCache

import torch
from utils.io import travel


class DiskCache(LRUCache):

    def __init__(self, max_size: int, root: str, fmt: str = '.pth'):
        super(DiskCache, self).__init__(max_size)
        self._fmt = fmt
        self._root = root if root.endswith('/') else root + '/'
        self._pf = f"{self._root}{{}}{self._fmt}"

    # 与原有lru衔接
    def __delitem__(self, key):
        if key in self:
            self.to_disk(key, self[key])
            super(DiskCache, self).__delitem__(key)

    def __missing__(self, key):
        try:
            self[key] = self.read_disk(key)
            return self[key]
        except:
            raise KeyError(key)

    def popitem(self):
        key, value = super(DiskCache, self).popitem()
        self.to_disk(key, value)
        return key, value

    # 新增硬盘操作
    def to_disk(self, key, value):
        dst = self._pf.format(key)
        parent, _ = os.path.split(dst)
        if not os.path.exists(parent):
            os.makedirs(parent)
        torch.save(value, dst)

    def read_disk(self, key):
        src = self._pf.format(key)
        if not os.path.exists(src):
            raise KeyError(f"No such cache entry on disk: {key}")
        return torch.load(src)

    def flush(self):
        for key in list(self.keys()):  # 使用list()复制keys视图以避免在迭代过程中修改字典
            self.to_disk(key, self[key])
        super(DiskCache, self).clear()

    def iter(self):
        for key in self:
            self.to_disk(key, self[key])
        for key in map(
                lambda x: x.removesuffix(self._fmt),
                filter(lambda x: x.endswith(self._fmt), travel(self._root))
        ):
            yield key, self[key]

    def delete(self, key):
        if key in self:
            super(DiskCache, self).__delitem__(key)
        src = self._pf.format(key)
        if os.path.exists(src):
            os.remove(src)

    def exists(self, key):
        if key in self:
            return True
        if os.path.exists(self._pf.format(key)):
            return True
        return False

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
