from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import pandas as pd
from Bio.Seq import Seq
from imblearn.under_sampling import RandomUnderSampler

from common.csv_sequence_windowing import window_sequences_parallel


class AbstractEmbedding(ABC):

    def __init__(self, min_size: int, max_size: int, overlap_percent: int):
        self.min_size = min_size
        self.max_size = max_size
        self.overlap_percent = overlap_percent

        self.x_resampled = None
        self.y_resampled = None
        self.y_train_aug = None
        self.x_train_aug = None
        self.y_windowed = None
        self.x_windowed = None
        self.df = None

    def load_data(self, file_path: str):
        self.df = pd.read_csv(file_path).dropna()

    def clean_data(self):
        self.df = self.df.dropna()

    def window_data(self):
        windowed_df = window_sequences_parallel(df=self.df, min_size=self.min_size, max_size=self.max_size,
                                                overlap_percent=self.overlap_percent)

        self.x_windowed = windowed_df['sequence'].values
        self.y_windowed = windowed_df['target'].values

    def augment_data(self):
        self.x_train_aug, self.y_train_aug = self.reverse_complement_augmentation(self.x_windowed, self.y_windowed)

    def resample_data(self):
        x_train_aug_df = pd.DataFrame({'sequence': self.x_train_aug})
        index_array = np.array(x_train_aug_df.index).reshape(-1, 1)  # Convert indexes to 2D array
        under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
        index_resampled, y_resampled = under_sampler.fit_resample(index_array, self.y_train_aug)
        self.x_resampled = x_train_aug_df.iloc[index_resampled.flatten()]['sequence'].values
        self.y_resampled = y_resampled

    @abstractmethod
    def encode_sequences(self, sequences: List[str]) -> pd.DataFrame:
        pass

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
