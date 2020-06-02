import datetime
import logging
import os
import sys

from config.config_loader import get_configs


class LoggingManager:
    @staticmethod
    def configure_logger() -> None:
        if not os.path.exists("log"):
            os.makedirs("log")

        now = datetime.datetime.now()
        filename = \
            f'log/{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{now.second}-' \
            f'{get_configs()["dataset"]["name"]}.log'
        logging_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(filename=filename, level=logging.INFO, format=logging_format)
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
