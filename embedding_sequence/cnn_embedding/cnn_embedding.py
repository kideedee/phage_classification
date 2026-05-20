import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from embedding_sequence.abstract_embedding import AbstractEmbedding


class CNNEmbeddingModel(nn.Module):
    def __init__(self, max_length, *args, **kwargs):
        # Conv1D layer with 64 filters, kernel size 6, ReLU activation
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv1d(4, 64, kernel_size=6, padding='same')
        self.relu = nn.ReLU()

        # MaxPooling1D with pool size 3
        self.pool = nn.MaxPool1d(3)

        # BatchNormalization
        self.bn1 = nn.BatchNorm1d(64)

        # Dropout (0.3)
        self.dropout = nn.Dropout(0.3)

        # Fully connected layers
        # Calculate feature size after pooling
        self.feature_size = max_length // 3  # After pooling with size 3

        self.fc1 = nn.Linear(64, 64)  # Using GlobalAveragePooling instead of flattening
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        # Apply convolution and activation
        x = self.relu(self.conv(x))

        # Apply pooling
        x = self.pool(x)

        # Apply batch normalization and dropout
        x = self.bn1(x)
        x = self.dropout(x)

        # Global average pooling (equivalent to GlobalAveragePooling1D in Keras)
        x = torch.mean(x, dim=2)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.bn2(x)

        return x


class CNNEmbedding(AbstractEmbedding):
    def __init__(self, data_dir, output_dir, min_size, max_size, overlap_percent, is_train, fold):
        super().__init__(
            embedding_type="cnn",
            data_dir=data_dir,
            output_dir=output_dir,
            min_size=min_size,
            max_size=max_size,
            overlap_percent=overlap_percent,
            is_train=is_train,
            fold=fold)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CNNEmbeddingModel(max_length=max_size)
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = 64

    def run(self, sequences: np.array, labels: np.array):
        sequences = sequences.tolist()

        all_features = []
        all_labels = []

        total_batches = (len(sequences) + self.batch_size - 1) // self.batch_size
        for i in tqdm(range(0, len(sequences), self.batch_size), desc="Processing batches", total=total_batches):
            batch_sequences = sequences[i:i + self.batch_size]
            batch_labels = labels[i:i + self.batch_size]

            batch_tensor = torch.stack([torch.tensor(seq, dtype=torch.float32) for seq in batch_sequences])
            batch_tensor = batch_tensor.to(self.device)  # Move data to GPU

            with torch.no_grad():
                features = self.model(batch_tensor)
                batch_features = features.cpu().numpy()

            all_features.extend(batch_features)
            all_labels.extend(batch_labels)

        return np.array(all_features), np.array(all_labels)
