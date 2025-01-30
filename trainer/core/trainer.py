import gc
from abc import abstractmethod
from traceback import format_exc
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter
from config.env import SEED, CACHE_SIZE, TB_OUTPUT
from utils.cache import DiskCache
from utils.logger import Logger
from ..util.metric import Metric
from utils.data.fed import FederatedDataset
from utils.format import print_banner
from utils.tool import set_seed, get_time


class FLTrainer:

    def __init__(self, model: Module, fds: FederatedDataset, **kwargs):
        self._fds = fds
        self._model = model
        self._parse_kwargs(**kwargs)
        print_banner(self.__class__.__name__)

    def _parse_kwargs(self, **kwargs):
        self.name = f"{self.__class__.__name__}{kwargs.get('tag', '')}"
        self.test_step = kwargs.get('test_step', 5)
        self.round = kwargs.get('round', 300)

    # 初始化
    def _init(self):
        set_seed(SEED, has_torch=True)
        self._writer = SummaryWriter(f'{TB_OUTPUT}/{self.name}')
        self._logger = Logger.get_logger(self.name, self._writer.log_dir)
        self._cache = DiskCache(CACHE_SIZE, f'{self._writer.log_dir}/run/{get_time()}')
        self._clients = list(self._fds)
        self._k = 0

    def _handle_metric(self, metric: Metric, tag: str, writer: SummaryWriter = None):
        suffix = writer.log_dir.split('/')[-1]
        self._logger.info(f"[{self._k}] {tag.capitalize()}{'' if suffix == self.name else f'({suffix})'}: {metric}")
        if writer is not None:
            writer.add_scalar(f'{tag}/acc', metric.acc, self._k)
            writer.add_scalar(f'{tag}/loss', metric.loss, self._k)
            writer.flush()

    def _step(self):
        self._k += 1

    def _clean(self):
        gc.collect()

    @abstractmethod
    def _state(self, cid):
        raise NotImplementedError

    @abstractmethod
    def _select_client(self):
        raise NotImplementedError

    @abstractmethod
    def _local_update(self, cids):
        raise NotImplementedError

    @abstractmethod
    def _aggregate(self, cids):
        raise NotImplementedError

    @abstractmethod
    def _val(self, cids):
        raise NotImplementedError

    @abstractmethod
    def _test(self):
        raise NotImplementedError

    def _train(self):
        # 1.选择参与设备
        selected = self._select_client()
        # 2.本地训练
        self._local_update(selected)
        # 3.聚合更新
        self._aggregate(selected)
        # 4.聚合模型验证
        self._val(selected)
        # 5.模型测试
        if self._k % self.test_step == 0:
            self._test()

    def start(self):
        self._init()
        try:
            while self._k <= self.round:
                self._train()
                self._step()
        except:
            self._logger.warning(format_exc())
        finally:
            self._clean()
