import logging
from datetime import datetime

from common.env_config import config

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# Logger cho thông tin chung
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Console handler
    ]
)
log = logging.getLogger(__name__)

# Logger riêng cho monitoring
monitor_logger = logging.getLogger('monitor')
monitor_logger.setLevel(logging.INFO)

# File handler
monitor_file_handler = logging.FileHandler(config.LOG_DIR / f"monitor_{current_time}.log", mode='w')
monitor_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
