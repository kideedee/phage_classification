import os
import pickle
from functools import lru_cache
from typing import List, Tuple, Optional

import numpy as np
import torch
from Bio import SeqIO
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from experiment.graph import overlap_graph_encoding
from experiment.graph.BipartiteData import BipartiteData


class OptimizedBioDataset(Dataset):
    def __init__(self, fasta_file: str, k: int = 3, transform=None,
                 cache_dir: Optional[str] = None, preload_sequences: bool = True,
                 max_sequences: int = 1000):
        super().__init__(transform)
        self.fasta_file = fasta_file
        self.k = k
        self.transform = transform
        self.cache_dir = cache_dir
        self.max_sequences = max_sequences

        # Pre-computed edge index (same for all samples)
        self._edge_index_cache = None

        # Initialize storage
        self.sequences_info = []
        self.labels = []
        self.sequences = []  # Store actual sequences if preloading

        # Initialize cache-related attributes
        self.feature_cache = {}
        self.cache_file = None

        # Setup caching first
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self._setup_cache()

        # Load data
        self._load_data(preload_sequences)

    def _load_data(self, preload_sequences: bool):
        """Load sequence metadata and optionally preload sequences."""
        print("Loading sequence data...")

        if preload_sequences:
            # Load all sequences into memory for fastest access
            for i, seq_record in enumerate(SeqIO.parse(self.fasta_file, "fasta")):
                if i >= self.max_sequences:
                    break

                self.sequences.append(str(seq_record.seq))
                self.sequences_info.append({
                    'id': seq_record.id,
                    'length': len(seq_record.seq),
                    'index': i
                })

                # Extract label
                if seq_record.id.split("_")[2] == 'temperate':
                    self.labels.append(0)
                else:
                    self.labels.append(1)
        else:
            # Only load metadata, keep file handle for later access
            self.sequences = None
            for i, seq_record in enumerate(SeqIO.parse(self.fasta_file, "fasta")):
                if i >= self.max_sequences:
                    break

                self.sequences_info.append({
                    'id': seq_record.id,
                    'length': len(seq_record.seq),
                    'index': i
                })

                if seq_record.id.split("_")[2] == 'temperate':
                    self.labels.append(0)
                else:
                    self.labels.append(1)

    def _setup_cache(self):
        """Setup feature caching system."""
        self.cache_file = os.path.join(self.cache_dir, f"features_k{self.k}.pkl")
        self.feature_cache = {}

        if os.path.exists(self.cache_file):
            print("Loading cached features...")
            with open(self.cache_file, 'rb') as f:
                self.feature_cache = pickle.load(f)

    def _save_cache(self):
        """Save feature cache to disk."""
        if self.cache_dir and self.feature_cache:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.feature_cache, f)

    @lru_cache(maxsize=1)
    def _get_edge_index(self):
        """Cached edge index computation."""
        if self._edge_index_cache is None:
            edge = []
            for i in range(4 ** (self.k * 2)):
                a = i // 4 ** self.k
                b = i % 4 ** self.k
                edge.append([a, i])
                edge.append([b, i])
            self._edge_index_cache = torch.tensor(np.array(edge).T, dtype=torch.long)
        return self._edge_index_cache

    def _get_sequence(self, idx: int) -> str:
        """Get sequence by index efficiently."""
        if self.sequences is not None:
            # Preloaded sequences
            return self.sequences[idx]
        else:
            # Load on demand (slower but memory efficient)
            sequences = list(SeqIO.parse(self.fasta_file, "fasta"))
            return str(sequences[idx].seq)

    def _compute_features(self, sequence: str, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute features for a sequence with caching."""
        cache_key = f"{idx}_{hash(sequence)}"

        # Check cache first
        if self.cache_dir and cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        # Compute features
        feature = overlap_graph_encoding.create_matrix_feature((sequence, self.k))

        pnode_feature = feature.reshape(-1, self.k - 1, 4 ** (self.k * 2))
        pnode_feature = np.moveaxis(pnode_feature, 1, 2)

        zero_layer = feature.reshape(-1, self.k - 1, 4 ** self.k, 4 ** self.k)[:, 0, :, :]
        fnode_feature = np.sum(zero_layer, axis=2).reshape(-1, 4 ** self.k, 1)

        x_p = torch.tensor(pnode_feature[0, :, :], dtype=torch.float)
        x_f = torch.tensor(fnode_feature[0, :, :], dtype=torch.float)

        # Cache results
        if self.cache_dir:
            self.feature_cache[cache_key] = (x_f, x_p)

        return x_f, x_p

    def __len__(self):
        return len(self.sequences_info)

    def __getitem__(self, idx):
        sequence = self._get_sequence(idx)
        label = self.labels[idx]

        # Get features (cached if available)
        x_f, x_p = self._compute_features(sequence, idx)

        # Get edge index (cached)
        edge_index = self._get_edge_index()

        y = torch.tensor([label], dtype=torch.long)

        data = BipartiteData(x_src=x_f, x_dst=x_p, edge_index=edge_index, y=y, num_nodes=None)

        if self.transform:
            data = self.transform(data)

        return data

    def get_batch(self, indices: List[int]) -> List[BipartiteData]:
        """Get multiple samples at once for batch processing."""
        batch_data = []

        for idx in indices:
            batch_data.append(self[idx])

        return batch_data

    def precompute_all_features(self, batch_size: int = 32, num_workers: int = 4):
        """Precompute all features in batches for maximum speed."""
        if not self.cache_dir:
            print("Warning: No cache directory specified. Features won't be saved.")
            return

        print(f"Precomputing features for {len(self)} sequences...")

        # Create batches of indices
        for start_idx in range(0, len(self), batch_size):
            end_idx = min(start_idx + batch_size, len(self))
            batch_indices = list(range(start_idx, end_idx))

            # Process batch
            for idx in batch_indices:
                if f"{idx}_{hash(self._get_sequence(idx))}" not in self.feature_cache:
                    sequence = self._get_sequence(idx)
                    self._compute_features(sequence, idx)

            # Save progress periodically
            if start_idx % (batch_size * 10) == 0:
                self._save_cache()
                print(f"Processed {end_idx}/{len(self)} sequences")

        # Final save
        self._save_cache()
        print("Feature precomputation completed!")

    def __del__(self):
        """Save cache when object is destroyed."""
        if hasattr(self, 'cache_dir') and self.cache_dir:
            self._save_cache()


class BatchBioDataLoader:
    """Optimized DataLoader wrapper for batch processing."""

    def __init__(self, dataset: OptimizedBioDataset, batch_size: int = 32,
                 shuffle: bool = True, num_workers: int = 4, pin_memory: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size

        # Use RTX 5070 Ti optimization
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,  # Faster GPU transfer
            persistent_workers=True if num_workers > 0 else False,  # Keep workers alive
            prefetch_factor=2 if num_workers > 0 else None,  # Prefetch batches
        )

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


# Usage example and performance optimization tips
def create_optimized_dataset(fasta_file: str, k: int = 3, cache_dir: str = "./cache") -> OptimizedBioDataset:
    """Create an optimized dataset with best practices."""

    # Option 1: Maximum speed (more memory usage)
    dataset = OptimizedBioDataset(
        fasta_file=fasta_file,
        k=k,
        cache_dir=cache_dir,
        preload_sequences=True,  # Load all sequences into RAM
        max_sequences=1000
    )

    # Precompute all features for maximum training speed
    dataset.precompute_all_features(batch_size=64, num_workers=4)

    return dataset


def create_memory_efficient_dataset(fasta_file: str, k: int = 3, cache_dir: str = "./cache") -> OptimizedBioDataset:
    """Create a memory-efficient dataset."""

    # Option 2: Memory efficient (slower but less RAM)
    dataset = OptimizedBioDataset(
        fasta_file=fasta_file,
        k=k,
        cache_dir=cache_dir,
        preload_sequences=False,  # Load sequences on demand
        max_sequences=1000
    )

    return dataset


# Example usage for training
def example_usage():
    """Example of how to use the optimized dataset."""

    # Create dataset
    dataset = create_optimized_dataset("E:\master\\final_project\data\my_data\\fasta\\100_400\\1\\train\data.fa", k=3)

    # Create optimized dataloader for RTX 5070 Ti
    dataloader = BatchBioDataLoader(
        dataset,
        batch_size=64,  # Adjust based on GPU memory
        shuffle=True,
        num_workers=8,  # RTX 5070 Ti can handle more workers
        pin_memory=True
    )

    # Training loop example
    for batch in dataloader:
        print(batch.size)
        # batch is already optimized for GPU transfer
        # Your training code here
        pass

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")


if __name__ == '__main__':
    example_usage()
