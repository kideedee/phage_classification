import os
from typing import List, Tuple, Dict, Literal

import numpy as np
import pandas as pd
import torch
from Bio.Seq import Seq
from gensim.models import Word2Vec
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from common.csv_sequence_windowing import window_sequences_parallel
from common.env_config import config
from logger.phg_cls_log import log

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


class DNASequenceDataset(Dataset):
    """Dataset class for DNA sequences for fine-tuning DNA-BERT."""

    def __init__(self, sequences, labels, tokenizer, max_length=512, kmer_size=6):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.kmer_size = kmer_size

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Format sequence as space-separated k-mers
        sequence = self.sequences[idx]
        kmers = [sequence[i:i + self.kmer_size] for i in range(len(sequence) - self.kmer_size + 1)]
        formatted_seq = " ".join(kmers)

        # Tokenize
        encoding = self.tokenizer(
            formatted_seq,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # Convert to correct format for Trainer
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


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


class DNASequenceEmbeddingDataset(Dataset):
    """Dataset cho việc embedding sequences với DNABERT-2."""

    def __init__(self, sequences, tokenizer, max_length=512):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # Thực hiện tokenize ngay trong __getitem__ để tận dụng đa luồng
        inputs = self.tokenizer(
            sequence,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        # Loại bỏ chiều batch
        return {k: v.squeeze(0) for k, v in inputs.items()}


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
        encoding_method: Method to encode DNA sequences ('word2vec', 'dna_bert', or 'dna_bert_2')
        kmer_size: Size of k-mers for sequence processing (for DNA-BERT)
        overlap_percent: Percentage of overlap for sequence windowing
        word2vec_model_path: Path to Word2Vec model file (if using word2vec)
        dna_bert_model_name: Name of pre-trained DNA-BERT or DNABERT-2 model
        dna_bert_pooling: Pooling strategy for DNA-BERT/DNABERT-2 ('cls' or 'mean')
        dna_bert_batch_size: Batch size for DNA-BERT/DNABERT-2 processing
        output_dir: Directory to save output files
        retrain_word2vec: Whether to force retraining of Word2Vec model
        is_fine_tune_dna_bert: Whether to fine-tune DNA-BERT/DNABERT-2 before embedding
        fine_tune_epochs: Number of epochs for fine-tuning
        fine_tune_batch_size: Batch size for fine-tuning
        fine_tune_learning_rate: Learning rate for fine-tuning
    """

    def __init__(
            self,
            encoding_method: Literal["word2vec", "dna_bert", "dna_bert_2", "one_hot"] = "word2vec",
            kmer_size: int = 6,
            overlap_percent: int = 50,
            min_size=100,
            max_size=400,
            word2vec_model_path: str = "phage_word2vec_model.bin",
            dna_bert_model_name: str = "zhihan1996/DNA_bert_6",
            dna_bert_pooling: Literal["cls", "mean"] = "cls",
            dna_bert_batch_size: int = 32,
            retrain_word2vec: bool = False,
            is_fine_tune_dna_bert: bool = False,
            fine_tune_epochs: int = 3,
            fine_tune_batch_size: int = 16,
            fine_tune_learning_rate: float = 5e-5,
            num_workers=2,
            prefetch_factor=2
    ):
        """
        Initialize the DNA sequence processor.

        Args:
            encoding_method: Method to encode DNA sequences ('word2vec', 'dna_bert', 'dna_bert_2', one_hot)
            kmer_size: Size of k-mers (note: for DNA-BERT should be 3, 4, 5, or 6)
            overlap_percent: Percentage of overlap for sequence windowing
            word2vec_model_path: Path to Word2Vec model file (if using word2vec)
            dna_bert_model_name: Name of pre-trained DNA-BERT or DNABERT-2 model
            dna_bert_pooling: Pooling strategy for DNA-BERT/DNABERT-2 ('cls' or 'mean')
            dna_bert_batch_size: Batch size for DNA-BERT/DNABERT-2 processing
            output_dir: Directory to save output files
            retrain_word2vec: Whether to force retraining of Word2Vec model
            is_fine_tune_dna_bert: Whether to fine-tune DNA-BERT/DNABERT-2 before embedding
            fine_tune_epochs: Number of epochs for fine-tuning
            fine_tune_batch_size: Batch size for fine-tuning
            fine_tune_learning_rate: Learning rate for fine-tuning
        """
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        self.encoding_method = encoding_method
        self.kmer_size = kmer_size
        self.min_size = min_size
        self.max_size = max_size
        self.overlap_percent = overlap_percent
        self.word2vec_model_path = word2vec_model_path
        self.dna_bert_model_name = dna_bert_model_name
        self.dna_bert_pooling = dna_bert_pooling
        self.dna_bert_batch_size = dna_bert_batch_size
        self.retrain_word2vec = retrain_word2vec
        self.is_fine_tune_dna_bert = is_fine_tune_dna_bert
        self.fine_tune_epochs = fine_tune_epochs
        self.fine_tune_batch_size = fine_tune_batch_size
        self.fine_tune_learning_rate = fine_tune_learning_rate

        # Will be initialized when needed
        self.word2vec_model = None
        self.dna_bert_tokenizer = None
        self.dna_bert_model = None

        if encoding_method == "one_hot":
            self.output_dir = os.path.join(config.MY_DATA_DIR, f"one_hot/{min_size}_{max_size}")
        elif encoding_method == "word2vec":
            self.output_dir = os.path.join(config.MY_DATA_DIR, f"word2vec/{min_size}_{max_size}")
        elif encoding_method == "dna_bert":
            self.output_dir = os.path.join(config.MY_DATA_DIR, f"dna_bert/{min_size}_{max_size}")
        elif encoding_method == "dna_bert_2":
            self.output_dir = os.path.join(config.MY_DATA_DIR, f"dna_bert_2/{min_size}_{max_size}")
        else:
            raise NotImplementedError

        # Validate parameters
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        if self.encoding_method not in ["word2vec", "dna_bert", "dna_bert_2", "one_hot"]:
            raise ValueError(
                f"Unknown encoding method: {self.encoding_method}. Choose 'word2vec', 'dna_bert', 'dna_bert_2', or 'one_hot'.")

        if self.encoding_method == "dna_bert" and self.kmer_size not in [3, 4, 5, 6]:
            raise ValueError("For DNA-BERT, kmer_size must be 3, 4, 5, or 6")

        # Ensure DNA-BERT model name matches kmer_size
        if self.encoding_method == "dna_bert" and str(self.kmer_size) not in self.dna_bert_model_name:
            log.info("Warning: kmer_size (%s) doesn't match DNA-BERT model name (%s)",
                     self.kmer_size, self.dna_bert_model_name)
            log.info("Updating model name to 'zhihan1996/DNA_bert_%s'", self.kmer_size)
            self.dna_bert_model_name = f"zhihan1996/DNA_bert_{self.kmer_size}"

        # For DNABERT-2, we keep the model name as specified
        if self.encoding_method == "dna_bert_2" and self.dna_bert_model_name != "zhihan1996/DNABERT-2-117M":
            log.info("Warning: Using custom DNABERT-2 model: %s", self.dna_bert_model_name)

        # Validate batch size
        if self.dna_bert_batch_size <= 0:
            raise ValueError(f"DNA-BERT/DNABERT-2 batch size must be positive, got {self.dna_bert_batch_size}")

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
        log.info("Loading DNA-BERT model: %s", self.dna_bert_model_name)
        self.dna_bert_tokenizer = AutoTokenizer.from_pretrained(self.dna_bert_model_name)
        self.dna_bert_model = AutoModel.from_pretrained(self.dna_bert_model_name)

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
        log.info("Loading DNABERT-2 model: %s", self.dna_bert_model_name)
        self.dna_bert_tokenizer = AutoTokenizer.from_pretrained(self.dna_bert_model_name)
        self.dna_bert_model = AutoModel.from_pretrained(self.dna_bert_model_name, trust_remote_code=True)

        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Using device for DNABERT-2: %s", device)
        self.dna_bert_model = self.dna_bert_model.to(device)

        return self.dna_bert_tokenizer, self.dna_bert_model

    def fine_tune_dna_bert(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
            output_model_dir: str = "fine_tuned_dna_bert",
            num_train_epochs: int = 3,
            per_device_train_batch_size: int = 16,
            weight_decay: float = 0.01,
            max_length: int = 512,
    ) -> str:
        """
        Fine-tune DNA-BERT for sequence classification.

        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            output_model_dir: Directory to save fine-tuned model
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size for training
            weight_decay: Weight decay for AdamW optimizer
            max_length: Maximum sequence length

        Returns:
            Path to fine-tuned model
        """
        if self.dna_bert_tokenizer is None or self.dna_bert_model is None:
            log.info("Loading DNA-BERT model for fine-tuning")
            self.load_dna_bert_model()

        # Add classification head to the model
        num_labels = len(np.unique(y_train))
        model = AutoModelForSequenceClassification.from_pretrained(
            self.dna_bert_model_name,
            num_labels=num_labels
        )

        # Move model to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Create datasets
        train_dataset = DNASequenceDataset(X_train, y_train, self.dna_bert_tokenizer, max_length, self.kmer_size)
        val_dataset = DNASequenceDataset(X_val, y_val, self.dna_bert_tokenizer, max_length, self.kmer_size)

        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.dna_bert_tokenizer)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_model_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size * 2,
            warmup_steps=500,
            weight_decay=weight_decay,
            learning_rate=self.fine_tune_learning_rate,
            logging_dir=f"{output_model_dir}/logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            fp16=torch.cuda.is_available(),  # Use mixed precision if available
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.dna_bert_tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        # Train the model
        log.info("Starting fine-tuning DNA-BERT")
        trainer.train()

        # Evaluate the model
        log.info("Evaluating fine-tuned model")
        eval_result = trainer.evaluate()
        log.info("Evaluation results: %s", eval_result)

        # Save model
        log.info("Saving fine-tuned model to %s", output_model_dir)
        trainer.save_model(output_model_dir)
        self.dna_bert_tokenizer.save_pretrained(output_model_dir)

        # Update model path
        self.dna_bert_model_name = output_model_dir

        # Reload model for embedding
        self.dna_bert_model = AutoModel.from_pretrained(output_model_dir)
        self.dna_bert_model = self.dna_bert_model.to(device)

        return output_model_dir

    def fine_tune_dna_bert_2(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
            output_model_dir: str = "fine_tuned_dna_bert_2",
            num_train_epochs: int = 3,
            per_device_train_batch_size: int = 16,
            weight_decay: float = 0.01,
            max_length: int = 512,
    ) -> str:
        """
        Fine-tune DNABERT-2 for sequence classification.

        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            output_model_dir: Directory to save fine-tuned model
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size for training
            weight_decay: Weight decay for AdamW optimizer
            max_length: Maximum sequence length

        Returns:
            Path to fine-tuned model
        """
        if self.dna_bert_tokenizer is None or self.dna_bert_model is None:
            log.info("Loading DNABERT-2 model for fine-tuning")
            self.load_dna_bert_2_model()

        # Add classification head to the model
        num_labels = len(np.unique(y_train))
        model = AutoModelForSequenceClassification.from_pretrained(
            self.dna_bert_model_name,
            num_labels=num_labels,
            trust_remote_code=True
        )

        # Move model to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Create datasets - DNABERT-2 directly uses sequences without k-mer formatting
        train_dataset = DNABERT2SequenceDataset(X_train, y_train, self.dna_bert_tokenizer, max_length)
        val_dataset = DNABERT2SequenceDataset(X_val, y_val, self.dna_bert_tokenizer, max_length)

        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.dna_bert_tokenizer)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_model_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size * 2,
            warmup_steps=500,
            weight_decay=weight_decay,
            learning_rate=self.fine_tune_learning_rate,
            logging_dir=f"{output_model_dir}/logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            fp16=torch.cuda.is_available(),  # Use mixed precision if available
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.dna_bert_tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        # Train the model
        log.info("Starting fine-tuning DNABERT-2")
        trainer.train()

        # Evaluate the model
        log.info("Evaluating fine-tuned model")
        eval_result = trainer.evaluate()
        log.info("Evaluation results: %s", eval_result)

        # Save model
        log.info("Saving fine-tuned model to %s", output_model_dir)
        trainer.save_model(output_model_dir)
        self.dna_bert_tokenizer.save_pretrained(output_model_dir)

        # Update model path
        self.dna_bert_model_name = output_model_dir

        # Reload model for embedding
        self.dna_bert_model = AutoModel.from_pretrained(output_model_dir)
        self.dna_bert_model = self.dna_bert_model.to(device)

        return output_model_dir

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
        if self.dna_bert_pooling == "cls":
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
        if self.dna_bert_pooling == "cls":
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
            batch_size = self.dna_bert_batch_size

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
            if self.dna_bert_pooling == "cls":
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
            self, sequences: np.ndarray, batch_size: int = None, max_length: int = 512
    ) -> np.ndarray:
        """Convert sequences to vectors using DNABERT-2 with explicit prefetching."""
        if self.dna_bert_tokenizer is None or self.dna_bert_model is None:
            raise ValueError("DNABERT-2 model not loaded.")

        if batch_size is None:
            batch_size = self.dna_bert_batch_size

        device = next(self.dna_bert_model.parameters()).device
        log.info("Using device: %s, batch size: %s", device, batch_size)

        # Create dataset
        dataset = DNASequenceEmbeddingDataset(
            sequences=sequences,
            tokenizer=self.dna_bert_tokenizer,
            max_length=max_length,
        )

        # Optimize DataLoader with explicit prefetch_factor
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

        # Store embedding dimension to ensure consistency
        embedding_dim = None
        result = []

        # Use tqdm to display progress
        pbar = tqdm(total=len(sequences), desc="Embedding sequences")

        # Set model to evaluation mode
        self.dna_bert_model.eval()

        # Use no_grad and autocast (if available) to optimize performance
        with torch.no_grad():
            if hasattr(torch.cuda, 'amp') and torch.cuda.is_available():
                from torch.cuda.amp import autocast
                context_manager = autocast()
            else:
                from contextlib import nullcontext
                context_manager = nullcontext()

            with context_manager:
                for batch_idx, batch_inputs in enumerate(dataloader):
                    # Move inputs to device
                    batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

                    outputs = self.dna_bert_model(**batch_inputs, output_hidden_states=True)

                    # Handle different output formats consistently
                    if isinstance(outputs, tuple):
                        # Handle tuple output format (may contain hidden states)
                        if len(outputs) > 1 and hasattr(outputs, 'hidden_states'):
                            hidden_states = outputs.hidden_states[-1]
                        else:
                            # First element is typically the last hidden state
                            hidden_states = outputs[0]
                    else:
                        # Handle object output format with hidden_states attribute
                        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                            hidden_states = outputs.hidden_states[-1]  # Get the last layer
                        else:
                            hidden_states = outputs.last_hidden_state

                    # Apply pooling strategy
                    if self.dna_bert_pooling == 'cls':
                        # Use [CLS] token embedding
                        batch_embeddings = hidden_states[:, 0].cpu().numpy()
                    elif self.dna_bert_pooling == 'mean':
                        # Average all token embeddings, ignoring padding
                        attention_mask = batch_inputs['attention_mask'].unsqueeze(-1)
                        batch_embeddings = torch.sum(hidden_states * attention_mask, 1) / torch.clamp(
                            torch.sum(attention_mask, 1), min=1e-9)
                        batch_embeddings = batch_embeddings.cpu().numpy()
                    elif self.dna_bert_pooling == 'max':
                        # Max pooling of token embeddings, ignoring padding
                        attention_mask = batch_inputs['attention_mask'].unsqueeze(-1)
                        masked_hidden = hidden_states * attention_mask
                        # Replace padding with large negative values
                        masked_hidden = masked_hidden.masked_fill((1 - attention_mask).bool(), -1e9)
                        batch_embeddings = torch.max(masked_hidden, dim=1)[0].cpu().numpy()

                    # Check and store embedding dimension on first batch
                    if embedding_dim is None:
                        embedding_dim = batch_embeddings.shape[1]
                        log.info(f"Embedding dimension: {embedding_dim}")

                    # Verify consistency with expected embedding dimensions
                    if batch_embeddings.shape[1] != embedding_dim:
                        log.warning(
                            f"Dimension mismatch at batch {batch_idx}: expected {embedding_dim}, got {batch_embeddings.shape[1]}")
                        # Ensure consistent dimensions by padding or truncating if necessary
                        if batch_embeddings.shape[1] < embedding_dim:
                            pad_width = ((0, 0), (0, embedding_dim - batch_embeddings.shape[1]))
                            batch_embeddings = np.pad(batch_embeddings, pad_width, mode='constant', constant_values=0)
                        else:
                            batch_embeddings = batch_embeddings[:, :embedding_dim]

                    # Add batch result to list
                    result.append(batch_embeddings)

                    # Update progress bar
                    pbar.update(batch_inputs["input_ids"].size(0))

        pbar.close()

        # Free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Stack all results
        return np.vstack(result)

    def encode_sequences(self, X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray = None,
                         y_val: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
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

        elif self.encoding_method == "dna_bert":
            log.info("Using DNA-BERT encoding method")

            # Load DNA-BERT model
            self.load_dna_bert_model()

            # Fine-tune DNA-BERT if requested
            if self.is_fine_tune_dna_bert:
                if y_train is None or y_val is None:
                    raise ValueError("Labels (y_train and y_val) are required for fine-tuning")

                log.info("Fine-tuning DNA-BERT model...")
                fine_tuned_model_dir = self.fine_tune_dna_bert(
                    max_length=self.max_size,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    output_model_dir=os.path.join(self.output_dir, "fine_tuned_dna_bert"),
                    num_train_epochs=self.fine_tune_epochs,
                    per_device_train_batch_size=self.fine_tune_batch_size
                )
                log.info("Fine-tuning complete! Model saved to %s", fine_tuned_model_dir)

            # Convert sequences to vectors using the batch_size from instance
            log.info("Using batch size of %s for DNA-BERT encoding", self.dna_bert_batch_size)
            X_train_vectors = self.convert_sequences_to_dna_bert_vectors(X_train)
            X_val_vectors = self.convert_sequences_to_dna_bert_vectors(X_val)

        elif self.encoding_method == "dna_bert_2":
            log.info("Using DNABERT-2 encoding method")

            # Load DNABERT-2 model
            self.load_dna_bert_2_model()

            # Fine-tune DNABERT-2 if requested
            if self.is_fine_tune_dna_bert:
                if y_train is None or y_val is None:
                    raise ValueError("Labels (y_train and y_val) are required for fine-tuning")

                log.info("Fine-tuning DNABERT-2 model...")
                fine_tuned_model_dir = self.fine_tune_dna_bert_2(
                    max_length=self.max_size,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    output_model_dir=os.path.join(self.output_dir, "fine_tuned_dna_bert_2"),
                    num_train_epochs=self.fine_tune_epochs,
                    per_device_train_batch_size=self.fine_tune_batch_size
                )
                log.info("Fine-tuning complete! Model saved to %s", fine_tuned_model_dir)

            # Convert sequences to vectors using the batch_size from instance
            log.info("Using batch size of %s for DNABERT-2 encoding", self.dna_bert_batch_size)
            X_train_vectors = self.convert_sequences_to_dna_bert_2_vectors(sequences=X_train, max_length=self.max_size)
            X_val_vectors = self.convert_sequences_to_dna_bert_2_vectors(sequences=X_val, max_length=self.max_size)
        elif self.encoding_method == "one_hot":
            # Sử dụng one-hot encoding từ DeePhage
            return self.encode_sequences_with_one_hot(X_train, X_val)
        else:
            raise ValueError(f"Unknown encoding method: {self.encoding_method}")

        return X_train_vectors, X_val_vectors

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
        # train_df = train_df.sample(100)
        # val_df = val_df.sample(10)

        # Step 2: Apply windowing and extract features
        log.info("Step 2: Applying sequence windowing with %s%% overlap", self.overlap_percent)
        x_train, y_train, x_val, y_val = self.window_and_extract_features(train_df, val_df)

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
        X_train_vectors, X_val_vectors = self.encode_sequences(
            X_train_resampled, x_val, y_train_resampled, y_val
        )

        # Step 6: Print shape information
        log.info("X_train_vectors shape: %s", X_train_vectors.shape)
        log.info("y_train_resampled shape: %s", y_train_resampled.shape)
        log.info("X_val_vectors shape: %s", X_val_vectors.shape)
        log.info("y_val shape: %s", y_val.shape)

        # Create output filename prefix
        if self.encoding_method == "word2vec":
            output_prefix = "word2vec"
        elif self.encoding_method == "one_hot":
            output_prefix = f"one_hot_{self.min_size}_{self.max_size}"
        elif self.encoding_method == "dna_bert":
            output_prefix = f"dna_bert_{self.kmer_size}_{self.dna_bert_pooling}"
            # Add fine-tuned to prefix if model was fine-tuned
            if self.is_fine_tune_dna_bert:
                output_prefix = f"fine_tuned_{output_prefix}"
        else:  # dna_bert_2
            output_prefix = f"dna_bert_2_{self.dna_bert_pooling}"
            # Add fine-tuned to prefix if model was fine-tuned
            if self.is_fine_tune_dna_bert:
                output_prefix = f"fine_tuned_{output_prefix}"

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


# Example usage
if __name__ == '__main__':
    # Example with Word2Vec encoding
    # length = "100_400"
    # min_length = 1200
    # max_length = 1800
    # output_model = f"phage_word2vec_model_{length}.bin"
    # output_dir = f"word2vec_output_{length}"

    # word2vec_processor = DNASequenceProcessor(
    #     encoding_method="word2vec",
    #     kmer_size=6,
    #     min_size=min_length,
    #     max_size=max_length,
    #     overlap_percent=30,
    #     word2vec_model_path=output_model,
    #     output_dir=output_dir,
    #     retrain_word2vec=False
    # )
    #
    # word2vec_processor.process(
    #     train_path=config.TRAIN_DATA_CSV_FILE,
    #     val_path=config.VAL_DATA_CSV_FILE
    # )

    # Example with DNA-BERT encoding and fine-tuning
    # dna_bert_processor = DNASequenceProcessor(
    #     encoding_method="dna_bert",
    #     kmer_size=6,  # Must be 3, 4, 5, or 6 for DNA-BERT
    #     overlap_percent=50,
    #     dna_bert_model_name="zhihan1996/DNA_bert_6",
    #     dna_bert_pooling="cls",
    #     dna_bert_batch_size=64,  # Reduced batch size due to GPU memory constraints during fine-tuning
    #     output_dir="../dna_bert_output",
    #     is_fine_tune_dna_bert=True,  # Enable fine-tuning
    #     fine_tune_epochs=3,
    #     fine_tune_batch_size=16,
    #     fine_tune_learning_rate=5e-5
    # )
    #
    # dna_bert_processor.process(
    #     train_path=config.TRAIN_DATA_CSV_FILE,
    #     val_path=config.VAL_DATA_CSV_FILE
    # )

    # Example with DNABERT-2 encoding and fine-tuning
    # dna_bert_2_processor = DNASequenceProcessor(
    #     min_size=100,
    #     max_size=400,
    #     encoding_method="dna_bert_2",
    #     overlap_percent=30,
    #     dna_bert_model_name="zhihan1996/DNABERT-2-117M",
    #     dna_bert_pooling="cls",
    #     dna_bert_batch_size=64,  # Adjust based on your GPU memory
    #     output_dir=f"dna_bert_2_output_{length}",
    #     is_fine_tune_dna_bert=True,  # Enable fine-tuning
    #     fine_tune_epochs=3,
    #     fine_tune_batch_size=16,
    #     fine_tune_learning_rate=5e-5,
    #     num_workers=2,
    #     prefetch_factor=2
    # )
    #
    # dna_bert_2_processor.process(
    #     train_path=config.TRAIN_DATA_CSV_FILE,
    #     val_path=config.VAL_DATA_CSV_FILE
    # )

    # Khởi tạo với one-hot encoding
    processor = DNASequenceProcessor(
        encoding_method="one_hot",
        min_size=1200,
        max_size=1800,
        overlap_percent=30
    )

    # Xử lý dữ liệu với one-hot encoding
    processor.process(
        train_path=config.TRAIN_DATA_CSV_FILE,
        val_path=config.VAL_DATA_CSV_FILE
    )
