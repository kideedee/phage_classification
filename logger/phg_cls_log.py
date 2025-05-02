import logging
import os

from common.env_config import config


def setup_logger(log_level=logging.INFO):
    # Logger riêng cho monitoring
    log = logging.getLogger('monitor')
    log.setLevel(log_level)

    stream_handler = logging.StreamHandler()
    log.addHandler(stream_handler)

    # File handler
    log_dir = os.path.join(config.LOG_DIR)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    monitor_file_handler = logging.FileHandler(os.path.join(log_dir, f"log.log"), mode='a')
    monitor_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

    log.addHandler(monitor_file_handler)

    return log


log = setup_logger()
