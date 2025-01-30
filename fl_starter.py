import sys
from os import environ
from traceback import format_exc
import ray
import hydra
import torchinfo
from omegaconf import DictConfig

from config.env import TB_OUTPUT
from utils.data import fed
from utils.logger import Logger
from benchmark.builder import build_federated_dataset, build_model, build_trainer

_logger = Logger.get_logger(__name__)


def setup_env():
    sys.argv.append(f'hydra.run.dir={TB_OUTPUT}')
    sys.argv.append(f'hydra/job_logging=none')
    sys.argv.append(f'hydra/hydra_logging=none')
    environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    environ['HYDRA_FULL_ERROR'] = '1'
    environ['OC_CAUSE'] = '1'


@hydra.main(config_path="config", config_name="rfl_cfg", version_base=None)
def run(cfg: DictConfig):
    fds = build_federated_dataset(cfg.fds.name, cfg.fds.args)
    fed.summary(fds)
    net = build_model(cfg.model.name, cfg.model.args)
    torchinfo.summary(net)
    ray.init(log_to_driver=False, num_gpus=1)
    for tn in cfg.trainer.names:
        try:
            trainer = build_trainer(tn, net, fds, cfg.trainer.args)
            trainer.start()
        except:
            _logger.info(format_exc())
    ray.shutdown()


if __name__ == '__main__':
    setup_env()
    run()
