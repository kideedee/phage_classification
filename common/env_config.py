import os

from dotenv import load_dotenv

load_dotenv()


# def get_project_root():
#     # Tìm vị trí của file .git hoặc requirements.txt để xác định root
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     while current_dir != os.path.dirname(current_dir):  # Không phải root của ổ đĩa
#         if os.path.exists(os.path.join(current_dir, '.git')) or \
#                 os.path.exists(os.path.join(current_dir, 'requirements.txt')):
#             return current_dir
#         current_dir = os.path.dirname(current_dir)
#     return current_dir  # Fallback


# Configuration
class Config:
    DEBUGGING = int(os.getenv("DEBUGGING", 0))
    DATA_DIR = os.getenv("DATA_DIR")
    PROJECT_DIR = os.getenv("PROJECT_DIR")

    CACHE_FOLDER = os.path.join(PROJECT_DIR, os.getenv("CACHE_FOLDER", "cache"))
    TEMP_FOLDER = os.path.join(PROJECT_DIR, os.getenv("TEMP_FOLDER", "temp"))
    LOG_DIR = os.path.join(PROJECT_DIR, os.getenv("LOG_DIR", "logs"))
    RESULT_DIR = os.path.join(PROJECT_DIR, os.getenv("RESULT_DIR", "result"))
    MODEL_DIR = os.path.join(PROJECT_DIR, os.getenv("MODEL_DIR", "model"))

    MY_DATA_DIR = os.path.join(DATA_DIR, "my_data")
    GEN_BANK_DIR = os.path.join(DATA_DIR, "gen_bank")
    CUSTOM_DATA_DIR = os.path.join(MY_DATA_DIR, "custom")  # csv file sequence
    DNA_BERT_2_DATA_DIR = os.path.join(MY_DATA_DIR, "dna_bert_2")
    DNA_BERT_2_TOKENIZER_DATA_DIR = os.path.join(MY_DATA_DIR, "dna_bert_2_tokenizer")
    PHATYP_FILTER_DNA_BERT_2_TOKENIZER_DATA_DIR = os.path.join(MY_DATA_DIR, "phatyp_filter_dna_bert_2_tokenizer")
    PHATYP_FILTER_DNA_BERT_S_TOKENIZER_DATA_DIR = os.path.join(MY_DATA_DIR, "phatyp_filter_dna_bert_s_tokenizer")
    ONE_HOT_DATA_DIR = os.path.join(MY_DATA_DIR, "one_hot")
    FCGR_DATA_DIR = os.path.join(MY_DATA_DIR, "new_embedding_imp/fcgr")
    VIT_PREPARED_DATA_DIR = os.path.join(MY_DATA_DIR, "vit_prepared_data")
    RESAMPLE_DATA_DIR = os.path.join(MY_DATA_DIR, "resample")
    TRAIN_DATA_FOLD_1_CSV_FILE = os.path.join(CUSTOM_DATA_DIR, os.getenv("TRAIN_DATA_FOLD_1_CSV_FILE"))
    TEST_DATA_FOLD_1_CSV_FILE = os.path.join(CUSTOM_DATA_DIR, os.getenv("TEST_DATA_FOLD_1_CSV_FILE"))
    TRAIN_DATA_FOLD_2_CSV_FILE = os.path.join(CUSTOM_DATA_DIR, os.getenv("TRAIN_DATA_FOLD_2_CSV_FILE"))
    TEST_DATA_FOLD_2_CSV_FILE = os.path.join(CUSTOM_DATA_DIR, os.getenv("TEST_DATA_FOLD_2_CSV_FILE"))
    TRAIN_DATA_FOLD_3_CSV_FILE = os.path.join(CUSTOM_DATA_DIR, os.getenv("TRAIN_DATA_FOLD_3_CSV_FILE"))
    TEST_DATA_FOLD_3_CSV_FILE = os.path.join(CUSTOM_DATA_DIR, os.getenv("TEST_DATA_FOLD_3_CSV_FILE"))
    TRAIN_DATA_FOLD_4_CSV_FILE = os.path.join(CUSTOM_DATA_DIR, os.getenv("TRAIN_DATA_FOLD_4_CSV_FILE"))
    TEST_DATA_FOLD_4_CSV_FILE = os.path.join(CUSTOM_DATA_DIR, os.getenv("TEST_DATA_FOLD_4_CSV_FILE"))
    TRAIN_DATA_FOLD_5_CSV_FILE = os.path.join(CUSTOM_DATA_DIR, os.getenv("TRAIN_DATA_FOLD_5_CSV_FILE"))
    TEST_DATA_FOLD_5_CSV_FILE = os.path.join(CUSTOM_DATA_DIR, os.getenv("TEST_DATA_FOLD_5_CSV_FILE"))
    FILTER_FROM_PHATYP_DATA = os.path.join(MY_DATA_DIR, os.getenv("FILTER_FROM_PHATYP_DATA"))

    DNA2VEC_MODEL_PATH = os.path.join(MODEL_DIR, "dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v")

    NCBI_DOWNLOAD_MAX_WORKERS = int(os.getenv("NCBI_DOWNLOAD_MAX_WORKERS", 4))
    NCBI_REQUEST_DELAY = float(os.getenv("NCBI_REQUEST_DELAY", 0.5))
    TRAIN_DATA_FASTA_FILE = os.path.join(DATA_DIR, os.getenv("TRAIN_DATA_FASTA_FILE"))
    VAL_DATA_FASTA_FILE = os.path.join(DATA_DIR, os.getenv("TEST_DATA_FASTA_FILE"))
    TRAIN_DATA_CSV_FILE = os.path.join(DATA_DIR, os.getenv("TRAIN_DATA_CSV_FILE"))
    VAL_DATA_CSV_FILE = os.path.join(DATA_DIR, os.getenv("VAL_DATA_CSV_FILE"))

    X_TRAIN_SMOTE = os.path.join(DATA_DIR, os.getenv("X_TRAIN_SMOTE"))
    Y_TRAIN_SMOTE = os.path.join(DATA_DIR, os.getenv("Y_TRAIN_SMOTE"))
    X_TRAIN_ADASYN = os.path.join(DATA_DIR, os.getenv("X_TRAIN_ADASYN"))
    Y_TRAIN_ADASYN = os.path.join(DATA_DIR, os.getenv("Y_TRAIN_ADASYN"))
    X_TRAIN_ENN = os.path.join(DATA_DIR, os.getenv("X_TRAIN_ENN"))
    Y_TRAIN_ENN = os.path.join(DATA_DIR, os.getenv("Y_TRAIN_ENN"))
    X_VAL = os.path.join(DATA_DIR, os.getenv("X_VAL"))
    Y_VAL = os.path.join(DATA_DIR, os.getenv("Y_VAL"))

    X_TRAIN_RUS_100_400 = os.path.join(DATA_DIR, os.getenv("X_TRAIN_RUS_100_400"))
    Y_TRAIN_RUS_100_400 = os.path.join(DATA_DIR, os.getenv("Y_TRAIN_RUS_100_400"))
    X_VAL_RUS_100_400 = os.path.join(DATA_DIR, os.getenv("X_VAL_RUS_100_400"))
    Y_VAL_RUS_100_400 = os.path.join(DATA_DIR, os.getenv("Y_VAL_RUS_100_400"))
    X_TRAIN_RUS_400_800 = os.path.join(DATA_DIR, os.getenv("X_TRAIN_RUS_400_800"))
    Y_TRAIN_RUS_400_800 = os.path.join(DATA_DIR, os.getenv("Y_TRAIN_RUS_400_800"))
    X_VAL_RUS_400_800 = os.path.join(DATA_DIR, os.getenv("X_VAL_RUS_400_800"))
    Y_VAL_RUS_400_800 = os.path.join(DATA_DIR, os.getenv("Y_VAL_RUS_400_800"))
    X_TRAIN_RUS_800_1200 = os.path.join(DATA_DIR, os.getenv("X_TRAIN_RUS_800_1200"))
    Y_TRAIN_RUS_800_1200 = os.path.join(DATA_DIR, os.getenv("Y_TRAIN_RUS_800_1200"))
    X_VAL_RUS_800_1200 = os.path.join(DATA_DIR, os.getenv("X_VAL_RUS_800_1200"))
    Y_VAL_RUS_800_1200 = os.path.join(DATA_DIR, os.getenv("Y_VAL_RUS_800_1200"))
    X_TRAIN_RUS_1200_1800 = os.path.join(DATA_DIR, os.getenv("X_TRAIN_RUS_1200_1800"))
    Y_TRAIN_RUS_1200_1800 = os.path.join(DATA_DIR, os.getenv("Y_TRAIN_RUS_1200_1800"))
    X_VAL_RUS_1200_1800 = os.path.join(DATA_DIR, os.getenv("X_VAL_RUS_1200_1800"))
    Y_VAL_RUS_1200_1800 = os.path.join(DATA_DIR, os.getenv("Y_VAL_RUS_1200_1800"))

    TRAIN_DNA_BERT_2_EMBEDDING = os.path.join(DATA_DIR, os.getenv("TRAIN_DNA_BERT_2_EMBEDDING"))
    TRAIN_DNA_BERT_2_LABELS = os.path.join(DATA_DIR, os.getenv("TRAIN_DNA_BERT_2_LABELS"))
    VAL_DNA_BERT_2_EMBEDDING = os.path.join(DATA_DIR, os.getenv("VAL_DNA_BERT_2_EMBEDDING"))
    VAL_DNA_BERT_2_LABELS = os.path.join(DATA_DIR, os.getenv("VAL_DNA_BERT_2_LABELS"))

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
