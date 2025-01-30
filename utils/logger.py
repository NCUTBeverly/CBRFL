import sys
from logging import Formatter, getLogger, FileHandler, StreamHandler
from config.env import LOG_LEVEL, LOG_FORMAT
from utils.tool import timestamp


class Logger:

    # Logger CONFIG
    LEVEL = LOG_LEVEL
    FORMATTER = Formatter(LOG_FORMAT)

    @classmethod
    def get_logger(cls, name, log_dir=None):
        logger = getLogger(name)
        logger.setLevel(cls.LEVEL)
        if log_dir is not None:
            file_handler = FileHandler(f'{log_dir}/{name}-{timestamp()}.log')
            file_handler.setFormatter(cls.FORMATTER)
            logger.addHandler(file_handler)
        console_handler = StreamHandler(sys.stdout)
        console_handler.setFormatter(cls.FORMATTER)
        logger.addHandler(console_handler)
        return logger
