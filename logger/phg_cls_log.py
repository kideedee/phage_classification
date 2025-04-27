import logging
import os
from datetime import datetime
from pathlib import Path
import inspect

from common.env_config import config

def setup_logger(script_path):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_name = os.path.basename(script_path).split('.')[0]

    # Logger cho thông tin chung
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()  # Console handler
        ]
    )
    # log = logging.getLogger(__name__)

    # Logger riêng cho monitoring
    log = logging.getLogger('monitor')
    log.setLevel(logging.INFO)

    # File handler
    log_dir = os.path.join(config.LOG_DIR, script_name)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    monitor_file_handler = logging.FileHandler(os.path.join(log_dir, f"{script_name}_{current_time}.log"), mode='w')
    monitor_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

    log.addHandler(monitor_file_handler)

    return log


log = setup_logger(inspect.getfile(inspect.currentframe()))
