import os

from dotenv import load_dotenv

load_dotenv()


def get_project_root():
    # Tìm vị trí của file .git hoặc requirements.txt để xác định root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != os.path.dirname(current_dir):  # Không phải root của ổ đĩa
        if os.path.exists(os.path.join(current_dir, '.git')) or \
                os.path.exists(os.path.join(current_dir, 'requirements.txt')):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return current_dir  # Fallback


# Configuration
class Config:
    CACHE_FOLDER = os.path.join(get_project_root(), os.getenv("CACHE_FOLDER", "cache"))
    if not os.path.exists(CACHE_FOLDER):
        os.makedirs(CACHE_FOLDER)
    TEMP_FOLDER = os.path.join(get_project_root(), os.getenv("TEMP_FOLDER", "temp"))
    if not os.path.exists(TEMP_FOLDER):
        os.makedirs(TEMP_FOLDER)
    LOG_DIR = os.path.join(get_project_root(), os.getenv("LOG_DIR", "logs"))
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    GEN_BANK_DIR = os.path.join(get_project_root(), os.getenv("GEN_BANK_DIR", "gen_bank"))
    if not os.path.exists(GEN_BANK_DIR):
        os.makedirs(GEN_BANK_DIR)

    NCBI_DOWNLOAD_MAX_WORKERS = int(os.getenv("NCBI_DOWNLOAD_MAX_WORKERS", 4))
    NCBI_REQUEST_DELAY = float(os.getenv("NCBI_REQUEST_DELAY", 0.5))

    INPUT_SIZE = int(os.getenv("INPUT_SIZE", 1024))
    HIDDEN_SIZE = int(os.getenv("HIDDEN_SIZE", 128))
    NUM_LAYERS = int(os.getenv("NUM_LAYERS", 2))
    NUM_HEADS = int(os.getenv("NUM_HEADS", 4))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 128))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.001))
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 100))
    BIDIRECTIONAL = os.getenv("BIDIRECTIONAL", "False").lower() in ("true", "1", "t")
    DROPOUT = float(os.getenv("DROPOUT", 0.3))
    WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 1e-4))
    LR_PATIENCE = int(os.getenv("LR_PATIENCE", 5))
    PATIENCE = int(os.getenv("PATIENCE", 20))

    # Paths
    DNA_BERT_INPUT_DATA_PATH = os.getenv("DNA_BERT_INPUT_DATA_PATH")
    DNA_BERT_OUTPUT_DATA_PATH = os.getenv("DNA_BERT_OUTPUT_DATA_PATH")
    DNA_BERT_MODEL_PATH = os.getenv("DNA_BERT_MODEL_PATH")


config = Config()
