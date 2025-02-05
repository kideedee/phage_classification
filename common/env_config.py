import os

from dotenv import load_dotenv

load_dotenv()


# Configuration
class Config:
    NUM_WORKERS = int(os.getenv("NUM_WORKERS", 8))
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
