import os.path
from abc import ABC, abstractmethod
from typing import List, Tuple, Any

import numpy as np
import pandas as pd
from Bio.Seq import Seq
from imblearn.under_sampling import RandomUnderSampler

from common.csv_sequence_windowing import window_sequences_parallel
from common.env_config import config
from logger.phg_cls_log import embedding_log as log


class AbstractEmbedding(ABC):

    def __init__(self, embedding_type, min_size: int, max_size: int, overlap_percent: int, is_train: bool, fold: int):
        self.embedding_type = embedding_type
        self.min_size = min_size
        self.max_size = max_size
        self.overlap_percent = overlap_percent
        self.fold = fold
        self.is_train = is_train
        self.log_prefix = "Embedding - "

        self.output_dir = None
        self.data_path = None
        self.post_construct()

    def post_construct(self):
        # Validate fold number
        if not (1 <= self.fold <= 5):
            raise ValueError("Invalid fold number")

        # Determine data source based on training/testing mode
        data_files = {
            True: {  # Training data files
                1: os.path.join(config.FILTER_FROM_PHATYP_DATA, f"{self.min_size}_{self.max_size}/{self.fold}/train/data.csv"),
                2: os.path.join(config.FILTER_FROM_PHATYP_DATA, f"{self.min_size}_{self.max_size}/{self.fold}/train/data.csv"),
                3: os.path.join(config.FILTER_FROM_PHATYP_DATA, f"{self.min_size}_{self.max_size}/{self.fold}/train/data.csv"),
                4: os.path.join(config.FILTER_FROM_PHATYP_DATA, f"{self.min_size}_{self.max_size}/{self.fold}/train/data.csv"),
                5: os.path.join(config.FILTER_FROM_PHATYP_DATA, f"{self.min_size}_{self.max_size}/{self.fold}/train/data.csv"),
            },
            False: {  # Test data files
                1: os.path.join(config.FILTER_FROM_PHATYP_DATA, f"{self.min_size}_{self.max_size}/{self.fold}/test/data.csv"),
                2: os.path.join(config.FILTER_FROM_PHATYP_DATA, f"{self.min_size}_{self.max_size}/{self.fold}/test/data.csv"),
                3: os.path.join(config.FILTER_FROM_PHATYP_DATA, f"{self.min_size}_{self.max_size}/{self.fold}/test/data.csv"),
                4: os.path.join(config.FILTER_FROM_PHATYP_DATA, f"{self.min_size}_{self.max_size}/{self.fold}/test/data.csv"),
                5: os.path.join(config.FILTER_FROM_PHATYP_DATA, f"{self.min_size}_{self.max_size}/{self.fold}/test/data.csv"),
            }
        }

        # Set data path based on training mode
        self.data_path = data_files[self.is_train][self.fold]

        # Set output directory (same for both train and test)
        mode_dir = "train" if self.is_train else "test"
        self.output_dir = os.path.join(
            config.MY_DATA_DIR,
            f"new_embedding_imp/{self.embedding_type}/{self.min_size}_{self.max_size}/fold_{self.fold}/{mode_dir}"
        )

    def load_data(self):
        log.info(f"{self.log_prefix}Loading data from {self.data_path}...")
        return pd.read_csv(self.data_path).dropna()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info(f"{self.log_prefix}Cleaning data...")
        return df.dropna()

    def window_data(self, df: pd.DataFrame) -> tuple[Any, Any]:
        windowed_df = window_sequences_parallel(df=df, min_size=self.min_size, max_size=self.max_size,
                                                overlap_percent=self.overlap_percent)

        return windowed_df['sequence'].values, windowed_df['target'].values

    def augment_data(self, x, y):
        x_aug, y_aug = self.reverse_complement_augmentation(x, y)
        return x_aug, y_aug

    def resample_data(self, x, y):
        x_train_aug_df = pd.DataFrame({'sequence': x})
        index_array = np.array(x_train_aug_df.index).reshape(-1, 1)  # Convert indexes to 2D array
        under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
        index_resampled, y_resampled = under_sampler.fit_resample(index_array, y)
        x_resampled = x_train_aug_df.iloc[index_resampled.flatten()]['sequence'].values
        y_resampled = y_resampled

        return x_resampled, y_resampled

    @abstractmethod
    def encode_sequences(self, sequences: List[str], labels: List[str]) -> tuple[
        np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]
    ]:
        pass

    # @abstractmethod
    def save_embedding(self, embeddings: np.array, labels: np.array) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

        data_dict = {
            f"{self.embedding_type}_vectors.npy": embeddings,
            f"{self.embedding_type}_labels.npy": labels,
        }

        for filename, data in data_dict.items():
            output_path = str(os.path.join(self.output_dir, filename))
            np.save(output_path, data)
            log.info("Saved %s with shape %s", filename, data.shape)

    @staticmethod
    def reverse_complement_augmentation(sequences: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
