import numpy as np
import torch
from torch.utils.data import Dataset


class ProteinEmbeddingDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray, is_training: bool = True):
        # Robust standardization using median and IQR
        median = np.median(sequences)
        q1 = np.percentile(sequences, 25)
        q3 = np.percentile(sequences, 75)
        iqr = q3 - q1

        # Clip extreme values and normalize
        sequences_clipped = np.clip(sequences, median - 3 * iqr, median + 3 * iqr)
        self.sequences = (sequences_clipped - median) / (iqr + 1e-8)
        self.sequences = self.sequences.reshape(-1, 1024, 1)
        self.labels = labels
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        sequence = torch.FloatTensor(self.sequences[idx])
        label = torch.tensor(self.labels[idx].item(), dtype=torch.float32)

        if self.is_training:
            # Add small random noise for regularization
            noise = torch.randn_like(sequence) * 0.01
            sequence = sequence + noise

        return sequence, label