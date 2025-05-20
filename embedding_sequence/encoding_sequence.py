import os
from collections import Counter
from typing import List, Tuple, Dict, Literal

import numpy as np
import pandas as pd
import torch
from Bio.Seq import Seq
from gensim.models import Word2Vec
from imblearn.under_sampling import RandomUnderSampler
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModel, BertForSequenceClassification
)

from common.csv_sequence_windowing import window_sequences_parallel
from common.env_config import config
from embedding_sequence import dna2vec
from logger.phg_cls_log import embedding_log as log

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def compute_metrics(pred):
    """Compute metrics for model evaluation."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


class DNABERT2SequenceDataset(Dataset):
    """Dataset class for DNA sequences for fine-tuning DNABERT-2."""

    def __init__(self, sequences, labels, tokenizer, max_length=512):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # DNABERT-2 can process raw sequences directly without k-mer formatting
        sequence = self.sequences[idx]

        # Tokenize
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # Convert to correct format for Trainer
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


class DNASequenceProcessor:
    """
    Class for processing DNA sequences with Word2Vec, DNA-BERT, or DNABERT-2 encoding.

    This class handles the complete pipeline for processing phage DNA sequences:
    1. Loading and cleaning data
    2. Windowing sequences
    3. Data augmentation
    4. Sequence encoding using Word2Vec, DNA-BERT, or DNABERT-2
    5. Saving processed data

    Attributes:
        encoding_method: Method to encode DNA sequences ('word2vec', 'dna_bert', 'dna2vec', or 'dna_bert_2')
        kmer_size: Size of k-mers for sequence processing (for DNA-BERT)
        overlap_percent: Percentage of overlap for sequence windowing
        word2vec_model_path: Path to Word2Vec model file (if using word2vec)
        dna_bert_2_tokenizer: Name of pre-trained DNA-BERT or DNABERT-2 model
        dna_bert_2_pooling: Pooling strategy for DNA-BERT/DNABERT-2 ('cls' or 'mean')
        dna_bert_2_batch_size: Batch size for DNA-BERT/DNABERT-2 processing
        output_dir: Directory to save output files
        retrain_word2vec: Whether to force retraining of Word2Vec model
    """

    def __init__(
            self,
            encoding_method: Literal["word2vec", "dna_bert_2", "dna2vec", "one_hot"] = "word2vec",
            kmer_size: int = 6,
            overlap_percent: int = 50,
            fold=1,
            min_size=100,
            max_size=400,
            dna_bert_2_tokenizer_path: str = "zhihan1996/DNA_bert_6",
            dna_bert_2_model_path: str = "zhihan1996/DNA_bert_6",
            dna_bert_pooling: Literal["cls", "mean"] = "cls",
            dna_bert_2_batch_size: int = 32,
            retrain_word2vec: bool = False,
            num_workers=2,
            prefetch_factor=2,

            dna2vec_method: Literal["average", "sum", "concat"] = "average",
    ):
        """
        Initialize the DNA sequence processor.

        Args:
            encoding_method: Method to encode DNA sequences ('word2vec', 'dna_bert_2', one_hot)
            kmer_size: Size of k-mers (note: for DNA-BERT should be 3, 4, 5, or 6)
            overlap_percent: Percentage of overlap for sequence windowing
            dna_bert_model_name: Name of pre-trained DNA-BERT or DNABERT-2 model
            dna_bert_pooling: Pooling strategy for DNA-BERT/DNABERT-2 ('cls' or 'mean')
            dna_bert_2_batch_size: Batch size for DNA-BERT/DNABERT-2 processing
            output_dir: Directory to save output files
            retrain_word2vec: Whether to force retraining of Word2Vec model
            is_fine_tune_dna_bert: Whether to fine-tune DNA-BERT/DNABERT-2 before embedding
            fine_tune_epochs: Number of epochs for fine-tuning
            fine_tune_batch_size: Batch size for fine-tuning
            fine_tune_learning_rate: Learning rate for fine-tuning
        """
        self.dna2vec_model = None
        self.prefetch_factor = prefetch_factor
        self.fold = fold
        self.num_workers = num_workers
        self.encoding_method = encoding_method
        self.kmer_size = kmer_size
        self.min_size = min_size
        self.max_size = max_size
        self.overlap_percent = overlap_percent
        self.dna_bert_2_tokenizer_path = dna_bert_2_tokenizer_path
        self.dna_bert_2_model_path = dna_bert_2_model_path
        self.dna_bert_2_pooling = dna_bert_pooling
        self.dna_bert_2_batch_size = dna_bert_2_batch_size
        self.retrain_word2vec = retrain_word2vec

        self.dna2vec_method = dna2vec_method

        # Will be initialized when needed
        self.word2vec_model = None
        self.dna_bert_tokenizer = None
        self.dna_bert_model = None
        self.dna_bert_2_model = None
        self.dna_bert_2_tokenizer = None

        if encoding_method == "one_hot":
            self.output_dir = os.path.join(config.MY_DATA_DIR, f"one_hot/{min_size}_{max_size}/{self.fold}")
        elif encoding_method == "word2vec":
            self.output_dir = os.path.join(config.MY_DATA_DIR, f"word2vec/{min_size}_{max_size}/{self.fold}")
            self.word2vec_model_path = os.path.join(config.MODEL_DIR,
                                                    f"word2vec/group_{min_size}_{max_size}/fold_{self.fold}/model.bin")
        elif encoding_method == "dna_bert":
            self.output_dir = os.path.join(config.MY_DATA_DIR, f"dna_bert/{min_size}_{max_size}/{self.fold}")
        elif encoding_method == "dna_bert_2":
            self.output_dir = os.path.join(config.MY_DATA_DIR, f"dna_bert_2/{min_size}_{max_size}/{self.fold}")
        elif encoding_method == "dna2vec":
            self.output_dir = os.path.join(config.MY_DATA_DIR, f"dna2vec/{min_size}_{max_size}/{self.fold}")
        else:
            raise NotImplementedError

        self.log_config()

        # Validate parameters
        self._validate_parameters()

    def log_config(self):
        """Log all configuration parameters."""
        import logging

        logger = logging.getLogger(__name__)

        logger.info("============ Configuration Parameters ============")
        logger.info(f"Encoding Method: {self.encoding_method}")
        logger.info(f"K-mer Size: {self.kmer_size}")
        logger.info(f"Overlap Percent: {self.overlap_percent}%")
        logger.info(f"Fold: {self.fold}")
        logger.info(f"Min Size: {self.min_size}")
        logger.info(f"Max Size: {self.max_size}")

        if self.encoding_method == "word2vec":
            logger.info(f"Word2Vec Model Path: {self.word2vec_model_path}")
            logger.info(f"Retrain Word2Vec: {self.retrain_word2vec}")

        elif self.encoding_method in ["dna_bert", "dna_bert_2"]:
            logger.info(f"DNA-BERT Model Name: {self.dna_bert_2_tokenizer}")
            logger.info(f"DNA-BERT Pooling: {self.dna_bert_2_pooling}")
            logger.info(f"DNA-BERT Batch Size: {self.dna_bert_2_batch_size}")

        elif self.encoding_method == "dna2vec":
            logger.info(f"DNA2Vec Method: {self.dna2vec_method}")

        logger.info(f"Number of Workers: {self.num_workers}")
        logger.info(f"Prefetch Factor: {self.prefetch_factor}")
        logger.info(f"Output Directory: {self.output_dir}")
        logger.info("================================================")

    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        if self.encoding_method not in ["word2vec", "dna_bert", "dna_bert_2", "dna2vec", "one_hot"]:
            raise ValueError(
                f"Unknown encoding method: {self.encoding_method}. Choose 'word2vec', 'dna_bert', 'dna_bert_2', 'dna2vec', or 'one_hot'.")

        if self.encoding_method == "dna_bert" and self.kmer_size not in [3, 4, 5, 6]:
            raise ValueError("For DNA-BERT, kmer_size must be 3, 4, 5, or 6")

        # Ensure DNA-BERT model name matches kmer_size
        if self.encoding_method == "dna_bert" and str(self.kmer_size) not in self.dna_bert_2_tokenizer:
            log.info("Warning: kmer_size (%s) doesn't match DNA-BERT model name (%s)",
                     self.kmer_size, self.dna_bert_2_tokenizer)
            log.info("Updating model name to 'zhihan1996/DNA_bert_%s'", self.kmer_size)
            self.dna_bert_2_tokenizer = f"zhihan1996/DNA_bert_{self.kmer_size}"

        # For DNABERT-2, we keep the model name as specified
        if self.encoding_method == "dna_bert_2" and self.dna_bert_2_tokenizer != "zhihan1996/DNABERT-2-117M":
            log.info("Warning: Using custom DNABERT-2 model: %s", self.dna_bert_2_tokenizer)

        # Validate batch size
        if self.dna_bert_2_batch_size <= 0:
            raise ValueError(f"DNA-BERT/DNABERT-2 batch size must be positive, got {self.dna_bert_2_batch_size}")

    def load_and_clean_data(self, train_path: str, val_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and validation data from CSV files and clean by removing NaN values.

        Args:
            train_path: Path to training CSV file
            val_path: Path to validation CSV file

        Returns:
            Tuple of cleaned training and validation DataFrames
        """
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

        log.info("Train DataFrame shape: %s", train_df.shape)
        log.info("Validation DataFrame shape: %s", val_df.shape)

        train_df = train_df.dropna()
        val_df = val_df.dropna()

        log.info("Train DataFrame shape after dropping NaN values: %s", train_df.shape)
        log.info("Validation DataFrame shape after dropping NaN values: %s", val_df.shape)

        return train_df, val_df

    def window_and_extract_features(
            self, train_df: pd.DataFrame, val_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply sequence windowing and extract features and labels.

        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame

        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        windowed_train_df = window_sequences_parallel(df=train_df, min_size=self.min_size, max_size=self.max_size,
                                                      overlap_percent=self.overlap_percent)
        windowed_val_df = window_sequences_parallel(df=val_df, min_size=self.min_size, max_size=self.max_size,
                                                    overlap_percent=self.overlap_percent)

        # Free memory
        del train_df, val_df

        X_train = windowed_train_df["sequence"].values
        y_train = windowed_train_df["target"].values
        X_val = windowed_val_df["sequence"].values
        y_val = windowed_val_df["target"].values

        # Free memory
        del windowed_train_df, windowed_val_df

        log.info("Number of phage sequences in training set: %s", len(X_train))
        log.info("Number of phage sequences in validation set: %s", len(X_val))

        return X_train, y_train, X_val, y_val

    def reverse_complement_augmentation(
            self, sequences: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create augmented data by adding reverse complement of DNA sequences.

        Args:
            sequences: Array of DNA sequences
            labels: Array of corresponding labels

        Returns:
            Tuple of augmented sequences and labels
        """
        augmented_sequences = []
        augmented_labels = []

        for seq, label in zip(sequences, labels):
            # Add original sequence
            augmented_sequences.append(seq)
            augmented_labels.append(label)

            # Add reverse complement sequence
            reverse_comp = str(Seq(seq).reverse_complement())
            augmented_sequences.append(reverse_comp)
            augmented_labels.append(label)

        return np.array(augmented_sequences), np.array(augmented_labels)

    def generate_kmers(self, sequence: str) -> List[str]:
        """
        Generate k-mers from a DNA sequence using sliding window.

        Args:
            sequence: DNA sequence string

        Returns:
            List of k-mers
        """
        return [sequence[i:i + self.kmer_size] for i in range(len(sequence) - self.kmer_size + 1)]

    def prepare_sequences_for_word2vec(self, sequences: np.ndarray) -> List[List[str]]:
        """
        Prepare sequences for Word2Vec by converting to valid k-mers.

        Args:
            sequences: Array of DNA sequences

        Returns:
            List of lists containing valid k-mers for each sequence
        """
        corpus = []
        valid_nucleotides = set("ACGT")

        for seq in sequences:
            # Create k-mers and keep only valid ones (containing only A, C, G, T)
            valid_kmers = [kmer for kmer in self.generate_kmers(seq)
                           if all(nucleotide in valid_nucleotides for nucleotide in kmer)]
            corpus.append(valid_kmers)

        return corpus

    def one_hot_encode_sequence(self, sequence: str) -> np.ndarray:
        """
        Mã hóa một chuỗi DNA theo định dạng one-hot encoding theo DeePhage.
        """
        # Chuẩn hóa chuỗi DNA (chuyển về chữ in hoa)
        sequence = sequence.upper()

        # Xác định độ dài tối đa
        max_length = self.max_size

        # Tạo ma trận kết quả (tất cả là 0)
        one_hot_matrix = np.zeros((max_length, 4), dtype=np.float32)

        # Tạo mapping từ nucleotide sang one-hot vector
        mapping = {
            'A': [0, 0, 0, 1],
            'C': [0, 0, 1, 0],
            'G': [0, 1, 0, 0],
            'T': [1, 0, 0, 0],
            'N': [0, 0, 0, 0]  # Xử lý trường hợp 'N'
        }

        # Điền ma trận với giá trị one-hot
        for i, nucleotide in enumerate(sequence[:max_length]):
            if nucleotide in mapping:
                one_hot_matrix[i] = mapping[nucleotide]

        return one_hot_matrix

    def one_hot_encode_sequences(self, X_train: np.ndarray, X_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mã hóa chuỗi DNA theo định dạng one-hot encoding.
        """
        # Mã hóa các chuỗi huấn luyện
        log.info("Encoding %s training sequences with one-hot encoding", len(X_train))
        X_train_encoded = np.array(
            [self.one_hot_encode_sequence(seq) for seq in tqdm(X_train, desc="Encoding training sequences")])

        # Mã hóa các chuỗi kiểm tra
        log.info("Encoding %s validation sequences with one-hot encoding", len(X_val))
        X_val_encoded = np.array(
            [self.one_hot_encode_sequence(seq) for seq in tqdm(X_val, desc="Encoding validation sequences")])

        return X_train_encoded, X_val_encoded

    def encode_sequences_with_one_hot(self, X_train: np.ndarray, X_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mã hóa chuỗi DNA sử dụng phương pháp one-hot encoding từ DeePhage.
        """
        log.info("Using one-hot encoding method from DeePhage")

        # Mã hóa với one-hot
        X_train_vectors, X_val_vectors = self.one_hot_encode_sequences(X_train, X_val)

        # In thông tin kích thước
        log.info("X_train_vectors shape: %s", X_train_vectors.shape)
        log.info("X_val_vectors shape: %s", X_val_vectors.shape)

        return X_train_vectors, X_val_vectors

    def train_word2vec_model(
            self, corpus: List[List[str]], vector_size: int = 300, window: int = 6, epochs: int = 20
    ) -> Word2Vec:
        """
        Train a Word2Vec model on DNA k-mers.

        Args:
            corpus: List of lists containing k-mers
            vector_size: Size of embedding vectors
            window: Context window size
            epochs: Number of training epochs

        Returns:
            Trained Word2Vec model
        """
        word2vec_model = Word2Vec(
            sentences=corpus,
            vector_size=vector_size,
            window=window,
            min_count=1,
            sample=1e-3,
            sg=1,  # Skip-gram model
            hs=0,  # Use negative sampling instead of hierarchical softmax
            epochs=20,
            negative=5,
            workers=10,
            seed=42
        )

        # Save model
        word2vec_model.save(self.word2vec_model_path)
        log.info("Word2Vec model saved to %s", self.word2vec_model_path)

        return word2vec_model

    def load_or_train_word2vec(self, sequences: np.ndarray) -> Word2Vec:
        """
        Load existing Word2Vec model or train new one if not present.

        Args:
            sequences: DNA sequences for training (if needed)

        Returns:
            Word2Vec model
        """
        if not self.retrain_word2vec and os.path.exists(self.word2vec_model_path):
            log.info("Loading existing Word2Vec model from %s", self.word2vec_model_path)
            self.word2vec_model = Word2Vec.load(self.word2vec_model_path)
        else:
            log.info("Training new Word2Vec model (will be saved to %s)", self.word2vec_model_path)
            corpus = self.prepare_sequences_for_word2vec(sequences)
            self.word2vec_model = self.train_word2vec_model(corpus)

        return self.word2vec_model

    def sequence_to_vector_word2vec(self, sequence: str) -> np.ndarray:
        """
        Convert a DNA sequence to feature vector using Word2Vec.

        Args:
            sequence: DNA sequence string

        Returns:
            Feature vector (numpy array)
        """
        if self.word2vec_model is None:
            raise ValueError("Word2Vec model not loaded. Call load_or_train_word2vec first.")

        kmers = self.generate_kmers(sequence)
        valid_kmers = [kmer for kmer in kmers if kmer in self.word2vec_model.wv.key_to_index]

        if not valid_kmers:
            return np.zeros(self.word2vec_model.vector_size)

        # Calculate average of vectors
        vectors = [self.word2vec_model.wv[kmer] for kmer in valid_kmers]
        return np.mean(vectors, axis=0)

    def convert_sequences_to_word2vec_vectors(self, sequences: np.ndarray) -> np.ndarray:
        """
        Convert all sequences to feature vectors using Word2Vec.

        Args:
            sequences: Array of DNA sequences

        Returns:
            Array of feature vectors
        """
        if self.word2vec_model is None:
            raise ValueError("Word2Vec model not loaded. Call load_or_train_word2vec first.")

        return np.array([self.sequence_to_vector_word2vec(seq) for seq in sequences])

    def load_dna_bert_model(self) -> Tuple[AutoTokenizer, AutoModel]:
        """
        Load DNA-BERT model and tokenizer.

        Returns:
            Tuple of (tokenizer, model)
        """
        log.info("Loading DNA-BERT model: %s", self.dna_bert_2_tokenizer)
        self.dna_bert_tokenizer = AutoTokenizer.from_pretrained(self.dna_bert_2_tokenizer)
        self.dna_bert_model = AutoModel.from_pretrained(self.dna_bert_2_tokenizer)

        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Using device for DNA-BERT: %s", device)
        self.dna_bert_model = self.dna_bert_model.to(device)

        return self.dna_bert_tokenizer, self.dna_bert_model

    def load_dna_bert_2_model(self) -> Tuple[AutoTokenizer, AutoModel]:
        """
        Load DNABERT-2 model and tokenizer.

        Returns:
            Tuple of (tokenizer, model)
        """
        log.info("Loading DNABERT-2 tokenizer: %s", self.dna_bert_2_tokenizer_path)
        log.info("Loading DNABERT-2 model: %s", self.dna_bert_2_model_path)
        self.dna_bert_2_tokenizer = AutoTokenizer.from_pretrained("../model/dna_bert_2/pretrained_100_400/1/tokenizer",
                                                                  trust_remote_code=True)
        self.dna_bert_2_model = BertForSequenceClassification.from_pretrained("../model/dna_bert_2/pretrained_100_400/1/finetune_dna_bert")

        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Using device for DNABERT-2: %s", device)
        self.dna_bert_2_model = self.dna_bert_2_model.to(device)

        return self.dna_bert_2_tokenizer, self.dna_bert_2_model

    def format_sequence_for_dna_bert(self, sequence: str) -> str:
        """
        Format DNA sequence for DNA-BERT by converting to k-mers separated by spaces.

        Args:
            sequence: DNA sequence string

        Returns:
            Space-separated k-mers string
        """
        kmers = self.generate_kmers(sequence)
        return " ".join(kmers)

    def dna_bert_encode_sequence(
            self, sequence: str, max_length: int = 512
    ) -> np.ndarray:
        """
        Encode a DNA sequence using DNA-BERT.

        Args:
            sequence: DNA sequence string
            max_length: Maximum sequence length for tokenizer

        Returns:
            Feature vector (numpy array)
        """
        if self.dna_bert_tokenizer is None or self.dna_bert_model is None:
            raise ValueError("DNA-BERT model not loaded. Call load_dna_bert_model first.")

        # Format sequence as space-separated k-mers
        formatted_seq = self.format_sequence_for_dna_bert(sequence)

        # Tokenize sequence
        inputs = self.dna_bert_tokenizer(
            formatted_seq,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )

        # Move inputs to the same device as model
        device = next(self.dna_bert_model.parameters()).device
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Get model outputs
        with torch.no_grad():
            outputs = self.dna_bert_model(**inputs)

        # Extract embeddings based on pooling strategy
        if self.dna_bert_2_pooling == "cls":
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        else:  # mean pooling
            # Use mean of all token embeddings (excluding padding)
            attention_mask = inputs["attention_mask"]
            last_hidden = outputs.last_hidden_state

            # Apply attention mask to get mean of non-padding tokens
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = (sum_embeddings / sum_mask).cpu().numpy()

        return embeddings[0]

    def dna_bert_2_encode_sequence(
            self, sequence: str, max_length: int = 512
    ) -> np.ndarray:
        """
        Encode a DNA sequence using DNABERT-2.

        Args:
            sequence: DNA sequence string
            max_length: Maximum sequence length for tokenizer

        Returns:
            Feature vector (numpy array)
        """
        if self.dna_bert_tokenizer is None or self.dna_bert_model is None:
            raise ValueError("DNABERT-2 model not loaded. Call load_dna_bert_2_model first.")

        # DNABERT-2 processes raw sequences directly
        # Tokenize sequence
        inputs = self.dna_bert_tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )

        # Move inputs to the same device as model
        device = next(self.dna_bert_model.parameters()).device
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # Get model outputs
        with torch.no_grad():
            outputs = self.dna_bert_model(**inputs)

        # Extract embeddings based on pooling strategy
        if self.dna_bert_2_pooling == "cls":
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        else:  # mean pooling
            # Use mean of all token embeddings (excluding padding)
            attention_mask = inputs["attention_mask"]
            last_hidden = outputs.last_hidden_state

            # Apply attention mask to get mean of non-padding tokens
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = (sum_embeddings / sum_mask).cpu().numpy()

        return embeddings[0]

    def convert_sequences_to_dna_bert_vectors(
            self, sequences: np.ndarray, batch_size: int = None, max_length: int = 512
    ) -> np.ndarray:
        """
        Convert all sequences to feature vectors using DNA-BERT in batches.

        Args:
            sequences: Array of DNA sequences
            batch_size: Batch size for processing (if None, use self.dna_bert_batch_size)
            max_length: Maximum sequence length for tokenizer

        Returns:
            Array of feature vectors
        """
        if self.dna_bert_tokenizer is None or self.dna_bert_model is None:
            raise ValueError("DNA-BERT model not loaded. Call load_dna_bert_model first.")

        # Use instance batch size if not explicitly provided
        if batch_size is None:
            batch_size = self.dna_bert_2_batch_size

        result = []
        device = next(self.dna_bert_model.parameters()).device
        log.info("Using device for batch processing: %s", device)
        log.info("Processing %s sequences with batch size %s", len(sequences), batch_size)

        for i in tqdm(range(0, len(sequences), batch_size), desc="Encoding sequences with DNA-BERT"):
            batch_seqs = sequences[i:i + batch_size]
            formatted_seqs = [self.format_sequence_for_dna_bert(seq) for seq in batch_seqs]

            # Tokenize batch
            inputs = self.dna_bert_tokenizer(
                formatted_seqs,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )

            # Move inputs to the same device as model
            inputs = {key: val.to(device) for key, val in inputs.items()}

            # Get model outputs
            with torch.no_grad():
                outputs = self.dna_bert_model(**inputs)

            # DNABERT-2 returns a tuple instead of an object with attributes
            # The first element in the tuple is typically the hidden states
            # Extract embeddings based on pooling strategy
            if self.dna_bert_2_pooling == "cls":
                # Use [CLS] token embedding (index 0)
                if isinstance(outputs, tuple):
                    last_hidden = outputs[0]  # First element of tuple contains hidden states
                else:
                    last_hidden = outputs.last_hidden_state
                batch_embeddings = last_hidden[:, 0, :].cpu().numpy()
            else:  # mean pooling
                # Use mean of all token embeddings (excluding padding)
                attention_mask = inputs["attention_mask"]
                if isinstance(outputs, tuple):
                    last_hidden = outputs[0]  # First element of tuple contains hidden states
                else:
                    last_hidden = outputs.last_hidden_state

                # Apply attention mask to get mean of non-padding tokens
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()

            result.append(batch_embeddings)

            # Log progress after every 10 batches
            if (i // batch_size) % 10 == 0 and i > 0:
                log.info("Processed %s/%s sequences", min(i + batch_size, len(sequences)), len(sequences))

        return np.vstack(result)

    def convert_sequences_to_dna_bert_2_vectors(
            self, sequences: np.ndarray, labels: np.ndarray, batch_size: int = None, max_length: int = 512
    ) -> (np.ndarray, np.ndarray):
        """Convert sequences to vectors using DNABERT-2 with preservation of all embedding dimensions."""
        if self.dna_bert_2_tokenizer is None or self.dna_bert_2_model is None:
            raise ValueError("DNABERT-2 model not loaded.")

        if batch_size is None:
            batch_size = self.dna_bert_2_batch_size

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Using device: %s, batch size: %s", device, batch_size)

        # Create dataset
        dataset = DNABERT2SequenceDataset(
            sequences=sequences,
            labels=labels,
            tokenizer=self.dna_bert_2_tokenizer,
            max_length=max_length,
        )

        # Optimize DataLoader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True,
            drop_last=False
        )

        # Khởi tạo mảng để lưu kết quả
        # Đầu tiên tìm kích thước embedding bằng cách chạy một mẫu
        with torch.no_grad():
            sample_inputs = self.dna_bert_2_tokenizer(
                sequences[0],
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding="max_length"
            ).to(device)

            sample_outputs = self.dna_bert_2_model(**sample_inputs, output_hidden_states=True)
            sample_hidden = sample_outputs.hidden_states[-1]

            if self.dna_bert_2_pooling == "cls":
                sample_emb = sample_hidden[:, 0, :].cpu().numpy()
            else:
                sample_emb = torch.mean(sample_hidden, dim=1).cpu().numpy()

            embedding_dim = sample_emb.shape[1]

        # Khởi tạo mảng kết quả với kích thước đầy đủ
        all_embeddings = np.zeros((len(sequences), embedding_dim), dtype=np.float32)
        all_labels = np.zeros(len(sequences), dtype=np.int32)

        # Xử lý từng batch và điền vào mảng kết quả
        current_idx = 0
        with tqdm(total=len(sequences), desc="Embedding sequences") as pbar:
            with torch.no_grad():
                for batch in dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    batch_labels = batch.pop('labels')
                    batch_size = batch_labels.size(0)

                    outputs = self.dna_bert_2_model(**batch, output_hidden_states=True)
                    last_hidden = outputs.hidden_states[-1]

                    if self.dna_bert_2_pooling == "cls":
                        batch_emb = last_hidden[:, 0, :].cpu().numpy()
                    elif self.dna_bert_2_pooling == "mean":
                        batch_emb = torch.mean(last_hidden, dim=1).cpu().numpy()
                    elif self.dna_bert_2_pooling == "max":
                        batch_emb = torch.max(last_hidden, dim=1).values.cpu().numpy()
                    else:
                        batch_emb = torch.min(last_hidden, dim=1).values.cpu().numpy()

                    # Điền kết quả vào mảng đã được khởi tạo trước
                    all_embeddings[current_idx:current_idx + batch_size] = batch_emb
                    all_labels[current_idx:current_idx + batch_size] = batch_labels.cpu().numpy()

                    current_idx += batch_size
                    pbar.update(batch_size)

        # Free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log.info(f"Final embedding shape: {all_embeddings.shape}")
        return all_embeddings, all_labels

    def encode_sequences(self, X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray = None,
                         y_val: np.ndarray = None):
        """
        Encode sequences using the selected method (Word2Vec, DNA-BERT, or DNABERT-2).
        Optionally fine-tune DNA-BERT/DNABERT-2 before encoding.

        Args:
            X_train: Training sequences
            X_val: Validation sequences
            y_train: Training labels (required for fine-tuning)
            y_val: Validation labels (required for fine-tuning)

        Returns:
            Tuple of (X_train_vectors, X_val_vectors)
        """
        if self.encoding_method == "word2vec":
            log.info("Using Word2Vec encoding method")

            # Load or train Word2Vec model
            self.load_or_train_word2vec(X_train)

            # Convert sequences to vectors
            X_train_vectors = self.convert_sequences_to_word2vec_vectors(X_train)
            X_val_vectors = self.convert_sequences_to_word2vec_vectors(X_val)

        elif self.encoding_method == "dna2vec":
            # log.info("Using DNA2Vec encoding method")
            # self.load_or_train_dna2vec(X_train)
            #
            # log.info("Start encoding sequences by dna2vec")
            # X_train_vectors, y_train_vectors = self.convert_sequences_to_dna2vec_vectors(X_train, y_train)
            # X_val_vectors, y_val_vectors = self.convert_sequences_to_dna2vec_vectors(X_val, y_val)
            # y_train = y_train_vectors
            # y_val = y_val_vectors
            raise NotImplementedError("Not implemented yet.")

        elif self.encoding_method == "dna_bert_2":
            log.info("Using DNABERT-2 encoding method")

            # Load DNABERT-2 model
            self.load_dna_bert_2_model()

            # Convert sequences to vectors using the batch_size from instance
            log.info("Using batch size of %s for DNABERT-2 encoding", self.dna_bert_2_batch_size)
            X_train_vectors, y_train = self.convert_sequences_to_dna_bert_2_vectors(sequences=X_train, labels=y_train,
                                                                                    max_length=self.max_size)
            X_val_vectors, y_val = self.convert_sequences_to_dna_bert_2_vectors(sequences=X_val, labels=y_val,
                                                                                max_length=self.max_size)
        elif self.encoding_method == "one_hot":
            # Sử dụng one-hot encoding từ DeePhage
            X_train_vectors, X_val_vectors = self.encode_sequences_with_one_hot(X_train, X_val)
        else:
            raise ValueError(f"Unknown encoding method: {self.encoding_method}")

        return X_train_vectors, y_train, X_val_vectors, y_val

    def save_processed_data(self, data_dict: dict) -> None:
        """
        Save processed data arrays to files.

        Args:
            data_dict: Dictionary mapping filenames to data arrays
        """
        os.makedirs(self.output_dir, exist_ok=True)

        for filename, data in data_dict.items():
            output_path = os.path.join(self.output_dir, filename)
            np.save(output_path, data)
            log.info("Saved %s with shape %s", filename, data.shape)

    def process(self, train_path: str, val_path: str) -> Dict[str, np.ndarray]:
        """
        Run the complete processing pipeline.

        Args:
            train_path: Path to training CSV file
            val_path: Path to validation CSV file

        Returns:
            Dictionary of processed data arrays
        """
        # Step 1: Load and clean data
        log.info("Step 1: Loading and cleaning data from %s and %s", train_path, val_path)
        train_df, val_df = self.load_and_clean_data(train_path, val_path)
        train_df = train_df.sample(10) # for testing
        val_df = val_df.sample(10) # for testing

        # Step 2: Apply windowing and extract features
        log.info("Step 2: Applying sequence windowing with %s%% overlap", self.overlap_percent)
        x_train, y_train, x_val, y_val = self.window_and_extract_features(train_df, val_df)
        counter = Counter(y_train)
        for k, v in counter.items():
            per = v / len(y_train) * 100
            log.info('Class=%d, Count=%d, Percentage=%.3f%%' % (k, v, per))
        log.info(f"x_val shape: {x_val.shape}")
        log.info(f"y_val shape: {y_val.shape}")

        # Step 3: Apply data augmentation
        log.info("Step 3: Applying reverse complement augmentation")
        X_train_aug, y_train_aug = self.reverse_complement_augmentation(x_train, y_train)
        log.info("Number of phage sequences in training set after augmentation: %s", len(X_train_aug))

        # Step 4: Apply random under-sampling
        log.info("Step 4: Apply random under-sampling")
        under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
        index_array = np.arange(len(X_train_aug)).reshape(-1, 1)
        index_resampled, y_train_resampled = under_sampler.fit_resample(index_array, y_train_aug)
        X_train_resampled = np.array([X_train_aug[i[0]] for i in index_resampled])

        # Step 5: Encode sequences with optional fine-tuning
        log.info("Step 5: Encoding sequences using %s method", self.encoding_method)
        # Pass labels for fine-tuning
        X_train_vectors, y_train, X_val_vectors, y_val = self.encode_sequences(
            X_train_resampled, x_val, y_train_resampled, y_val
        )

        # Step 6: Print shape information
        log.info("X_train_vectors shape: %s", X_train_vectors.shape)
        log.info("y_train_resampled shape: %s", y_train_resampled.shape)
        log.info("X_val_vectors shape: %s", X_val_vectors.shape)
        log.info("y_val shape: %s", y_val.shape)

        # Create output filename prefix
        if self.encoding_method == "word2vec":
            output_prefix = f"word2vec"
        elif self.encoding_method == "dna2vec":
            output_prefix = f"dna2vec_{self.min_size}_{self.max_size}"
        elif self.encoding_method == "one_hot":
            output_prefix = f"one_hot_{self.min_size}_{self.max_size}"
        elif self.encoding_method == "dna_bert":
            output_prefix = f"dna_bert_{self.kmer_size}_{self.dna_bert_2_pooling}"
        else:  # dna_bert_2
            output_prefix = f"dna_bert_2_{self.dna_bert_2_pooling}"

        # Step 7: Save processed data
        log.info("Step 7: Saving processed data to %s", self.output_dir)
        processed_data = {
            f"{output_prefix}_train_vector.npy": X_train_vectors,
            "y_train.npy": y_train_resampled,
            f"{output_prefix}_val_vector.npy": X_val_vectors,
            "y_val.npy": y_val
        }

        self.save_processed_data(processed_data)
        log.info("Processing complete!")

        return processed_data

    def load_or_train_dna2vec(self, X_train):
        self.dna2vec_model = dna2vec.MultiKModel(config.DNA2VEC_MODEL_PATH)

    def convert_sequences_to_dna2vec_vectors(self, sequences: np.ndarray, label: np.array) -> tuple:
        """Chuyển đổi các chuỗi DNA thành các vector sử dụng dna2vec với xử lý song song.
        Trả về vectors và labels đã được lọc (loại bỏ các None)."""

        # Tạo một bản sao của model để tránh chia sẻ đối tượng phức tạp giữa các quy trình
        # Đặt làm giá trị mặc định cho tham số của hàm bên trong
        dna2vec_model = self.dna2vec_model
        dna2vec_method = self.dna2vec_method

        # Tính số lượng batch dựa trên số lõi CPU
        num_cores = os.cpu_count()
        batch_size = max(1, len(sequences) // (num_cores * 4))  # Đảm bảo mỗi CPU xử lý nhiều batch

        # Chia dữ liệu thành các batch để giảm chi phí xử lý song song
        # Lưu lại các index gốc để phân biệt sequences nào là None
        batches = []
        batch_indices = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            indices = list(range(i, min(i + batch_size, len(sequences))))
            batches.append(batch)
            batch_indices.append(indices)

        # Hàm xử lý một batch sequences, với context và tham số được truyền vào
        def process_batch(batch, indices, model=dna2vec_model, method=dna2vec_method):
            batch_results = []
            valid_indices = []
            for i, seq in enumerate(batch):
                result = embed_sequence_by_dna2vec(seq, model, method)
                if result is not None:
                    batch_results.append(result)
                    valid_indices.append(indices[i])
            return batch_results, valid_indices

        # Hàm embedding tách ra khỏi class để tránh truyền self vào quá trình song song
        def embed_sequence_by_dna2vec(sequence, model, method, k_min=3, k_max=8):
            try:
                # Bỏ log debug cho mỗi sequence để tăng hiệu suất
                sequence = ''.join([c for c in sequence.upper() if c in 'ACGT'])

                # Tạo các k-mer từ chuỗi DNA
                kmers = []
                for k in range(k_min, k_max + 1):
                    if k > len(sequence):
                        continue
                    kmers.extend([sequence[i:i + k] for i in range(len(sequence) - k + 1)])

                # Lấy vector cho từng k-mer
                kmer_vectors = []
                for kmer in kmers:
                    try:
                        vec = model.vector(kmer)
                        kmer_vectors.append(vec)
                    except KeyError:
                        continue

                # Tổng hợp các vector
                if not kmer_vectors:
                    return None

                if method == 'average':
                    return np.mean(kmer_vectors, axis=0)
                elif method == 'sum':
                    return np.sum(kmer_vectors, axis=0)
                elif method == 'concat':
                    max_kmers = 100
                    selected_kmers = kmer_vectors[:max_kmers]
                    return np.concatenate(selected_kmers)
                else:
                    return None
            except Exception:
                # Bỏ log lỗi chi tiết để tránh tranh chấp khi ghi log
                return None

        # Sử dụng tqdm bên ngoài Parallel để hiển thị tiến trình tổng thể
        log.info(f"Processing {len(sequences)} sequences with {len(batches)} batches by {num_cores} cores")

        # Cấu hình Joblib để tối ưu hóa cho CPU
        results = Parallel(
            n_jobs=num_cores,  # Sử dụng số lõi tối đa
            verbose=10,  # Tăng verbosity để thấy tiến trình rõ hơn
            batch_size=1,  # Mỗi worker xử lý 1 batch một lần
            prefer="threads",  # Sử dụng thread thay vì processes nếu model có thể chia sẻ
            backend="loky",  # Backend hiệu quả cho Python object serialization
            timeout=None,  # Không giới hạn thời gian
            max_nbytes=None  # Không giới hạn mem_map cho dữ liệu lớn
        )(delayed(process_batch)(batch, indices) for batch, indices in zip(batches, batch_indices))

        # Làm phẳng các kết quả từ các batch và lưu lại các chỉ số hợp lệ
        all_results = []
        valid_indices = []
        for batch_result, batch_valid_indices in results:
            all_results.extend(batch_result)
            valid_indices.extend(batch_valid_indices)

        log.info(f"Completed embedding: {len(all_results)}/{len(sequences)} sequences")

        # Trả về numpy array vectors và labels tương ứng
        if not all_results:
            return np.array([]), np.array([])

        # Lọc labels dựa trên các chỉ số hợp lệ
        filtered_labels = label[valid_indices] if label is not None else None

        return np.array(all_results), filtered_labels
