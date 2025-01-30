import logging
from torch.multiprocessing import set_sharing_strategy

set_sharing_strategy('file_system')
# 日志
LOG_LEVEL = logging.INFO
LOG_FORMAT = '[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'

# 性能超参数
SEED = 2077
CACHE_SIZE = 10
ACTOR_NUM = 6
WORKER_NUM = 1
PREFETCH_FACTOR = 1

# 挂在盘符
D = '/mnt/d/'

# 项目地址
PROJECT = f'{D}/project/python/3l'
# 数据集
DATASET = f'/{D}/dataset'
LEAF = f'{DATASET}/leaf'
# 项目输出
TB_OUTPUT = f'{PROJECT}/asset/output/robust/slf'

