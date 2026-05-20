import h5py
import torch
from torch.utils.data import Dataset


class TwoTowerDataset(Dataset):
    def __init__(self, cnn_path, dna_bert_2_path):
        self.cnn_path = cnn_path
        self.dna_bert_2_path = dna_bert_2_path
        with h5py.File(self.cnn_path, 'r') as f:
            self.length = len(f['labels'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Open both files in a single context
        with h5py.File(self.cnn_path, 'r') as cnn_f, h5py.File(self.dna_bert_2_path, 'r') as dna_f:
            # Load and convert both vectors
            cnn_vector = torch.from_numpy(cnn_f['vectors'][idx])
            dna_vector = torch.from_numpy(dna_f['vectors'][idx])
            label = cnn_f['labels'][idx]

            # Concatenate and return
            return torch.cat([cnn_vector, dna_vector], dim=-1), label


class TwoTowerDatasetV2(Dataset):
    def __init__(self, one_hot_path, dna_bert_2_path):
        self.one_hot_path = one_hot_path
        self.dna_bert_2_path = dna_bert_2_path
        with h5py.File(self.one_hot_path, 'r') as f:
            self.length = len(f['labels'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Open both files in a single context
        with h5py.File(self.one_hot_path, 'r') as cnn_f, h5py.File(self.dna_bert_2_path, 'r') as dna_f:
            # Load and convert both vectors
            one_hot_vec = torch.from_numpy(cnn_f['vectors'][idx])
            bert_vec = torch.from_numpy(dna_f['vectors'][idx])
            label = cnn_f['labels'][idx]

            # Concatenate and return
            return one_hot_vec, bert_vec, label
