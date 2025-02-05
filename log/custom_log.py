import logging
from datetime import datetime
from pathlib import Path

# Tạo thư mục logs nếu chưa tồn tại
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
monitor_file = log_dir / f"gpu_monitor_{current_time}.json"

# Logger cho thông tin chung
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Console handler
    ]
)
logger = logging.getLogger(__name__)

# Logger riêng cho monitoring
monitor_logger = logging.getLogger('monitor')
monitor_logger.setLevel(logging.INFO)

# File handler
monitor_file_handler = logging.FileHandler(log_dir / f"monitor_{current_time}.log", mode='w')
monitor_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
