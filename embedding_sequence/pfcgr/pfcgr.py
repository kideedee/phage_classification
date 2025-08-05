"""
Positional Frequency Chaos Game Representation (PFCGR) Implementation
Extension of FCGR to include positional information via statistical moments

Based on the paper: "Positional frequency chaos game representation for machine learning-based 
classification of crop lncRNAs" by Papastathopoulos-Katsaros et al. (2025)

This implementation extends the traditional FCGR by adding four statistical measures 
of k-mer positions: mean, standard deviation, skewness, and kurtosis.

Enhanced version with improved error handling, validation, and additional features.
"""

import warnings
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Literal

import numpy as np
from scipy import stats


class PFCGR:
    def __init__(self, k: int = 6):
        if not 1 <= k <= 10:
            raise ValueError(f"k must be between 1 and 10, got {k}")

        self.k = k
        self.grid_size = 2 ** k
        self.nucleotides = ['A', 'C', 'G', 'T']

        # Create coordinate mapping for nucleotides
        self.coord_map = {
            'A': (0, 0), 'C': (0, 1),
            'G': (1, 0), 'T': (1, 1)
        }

        # Pre-generate k-mer positions for efficiency
        self._kmer_positions = self._generate_kmer_positions()

    def _generate_kmer_positions(self) -> Dict[str, Tuple[int, int]]:
        positions = {}

        def generate_all_kmers(length):
            """Generate all possible k-mers of given length"""
            if length == 1:
                return self.nucleotides
            else:
                prev_kmers = generate_all_kmers(length - 1)
                return [kmer + nuc for kmer in prev_kmers for nuc in self.nucleotides]

        all_kmers = generate_all_kmers(self.k)

        for kmer in all_kmers:
            row, col = 0, 0
            for i, nucleotide in enumerate(kmer):
                coord = self.coord_map[nucleotide]
                row += coord[0] * (2 ** (self.k - 1 - i))
                col += coord[1] * (2 ** (self.k - 1 - i))
            positions[kmer] = (row, col)

        return positions

    def _extract_kmers_with_positions(self, sequence: str) -> Dict[str, List[int]]:
        # Clean sequence: uppercase and remove non-ACGT characters
        sequence = sequence.upper()
        cleaned_sequence = ''.join(c for c in sequence if c in self.nucleotides)

        if len(cleaned_sequence) < self.k:
            warnings.warn(f"Sequence length ({len(cleaned_sequence)}) is shorter than k-mer size ({self.k})")
            return {}

        kmer_positions = defaultdict(list)

        for i in range(len(cleaned_sequence) - self.k + 1):
            kmer = cleaned_sequence[i:i + self.k]
            kmer_positions[kmer].append(i)

        return dict(kmer_positions)

    def _calculate_statistics(self, positions: List[int]) -> Tuple[float, float, float, float]:
        if not positions:
            return 0.0, 0.0, 0.0, 0.0

        n = len(positions)
        if n == 1:
            return float(positions[0]), 0.0, 0.0, 0.0

        # Use numpy for statistical calculations
        pos_array = np.array(positions, dtype=np.float64)

        mean = np.mean(pos_array)
        std_dev = np.std(pos_array, ddof=1) if n > 1 else 0.0

        # Calculate skewness and kurtosis with proper handling
        try:
            skewness = stats.skew(pos_array) if n > 2 else 0.0
            kurt = stats.kurtosis(pos_array) if n > 3 else 0.0
        except:
            # Handle any numerical issues
            skewness, kurt = 0.0, 0.0

        return mean, std_dev, skewness, kurt

    def _normalize_matrix(self, matrix: np.ndarray,
                          method: Literal['minmax', 'zscore', 'none'] = 'minmax') -> np.ndarray:
        if method == 'none':
            return matrix

        if method == 'minmax':
            if matrix.max() == matrix.min():
                return matrix
            return (matrix - matrix.min()) / (matrix.max() - matrix.min())

        elif method == 'zscore':
            if matrix.std() == 0:
                return matrix - matrix.mean()
            return (matrix - matrix.mean()) / matrix.std()

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def generate_fcgr(self, sequence: str, normalize: bool = True) -> np.ndarray:
        if not sequence:
            raise ValueError("Sequence cannot be empty")

        kmer_counts = defaultdict(int)

        # Count k-mers
        sequence = sequence.upper()
        cleaned_sequence = ''.join(c for c in sequence if c in self.nucleotides)

        for i in range(len(cleaned_sequence) - self.k + 1):
            kmer = cleaned_sequence[i:i + self.k]
            kmer_counts[kmer] += 1

        # Create FCGR matrix
        fcgr_matrix = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        for kmer, count in kmer_counts.items():
            if kmer in self._kmer_positions:
                row, col = self._kmer_positions[kmer]
                fcgr_matrix[row, col] = count

        if normalize:
            fcgr_matrix = self._normalize_matrix(fcgr_matrix)

        return fcgr_matrix

    def generate_pfcgr(self,
                       sequence: str,
                       return_format: Literal['channels', 'separate', 'tabular'] = 'channels',
                       normalization: Literal['minmax', 'zscore', 'none'] = 'minmax',
                       include_mask: bool = False) -> Union[np.ndarray, Dict[str, np.ndarray], Dict[str, float]]:
        if not sequence:
            raise ValueError("Sequence cannot be empty")

        if return_format not in ['channels', 'separate', 'tabular']:
            raise ValueError(f"Invalid return_format: {return_format}. Must be 'channels', 'separate', or 'tabular'")

        kmer_with_positions = self._extract_kmers_with_positions(sequence)

        if not kmer_with_positions:
            warnings.warn("No valid k-mers found in sequence")

        # Initialize matrices for each statistical measure
        freq_matrix = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        mean_matrix = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        std_matrix = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        skew_matrix = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        kurt_matrix = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Optional mask matrix
        if include_mask and return_format == 'channels':
            mask_matrix = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Fill matrices
        for kmer, positions in kmer_with_positions.items():
            if kmer in self._kmer_positions:
                row, col = self._kmer_positions[kmer]

                # Frequency (traditional FCGR)
                freq_matrix[row, col] = len(positions)

                # Statistical moments
                mean, std_dev, skewness, kurtosis = self._calculate_statistics(positions)
                mean_matrix[row, col] = mean
                std_matrix[row, col] = std_dev
                skew_matrix[row, col] = skewness
                kurt_matrix[row, col] = kurtosis

                # Mark presence in mask
                if include_mask and return_format == 'channels':
                    mask_matrix[row, col] = 1.0

        # Normalize all matrices
        freq_matrix = self._normalize_matrix(freq_matrix, normalization)
        mean_matrix = self._normalize_matrix(mean_matrix, normalization)
        std_matrix = self._normalize_matrix(std_matrix, normalization)
        skew_matrix = self._normalize_matrix(skew_matrix, normalization)
        kurt_matrix = self._normalize_matrix(kurt_matrix, normalization)

        if return_format == 'channels':
            # Return as multi-channel array (C, H, W) format for CNN
            channels = [freq_matrix, mean_matrix, std_matrix, skew_matrix, kurt_matrix]
            if include_mask:
                channels.append(mask_matrix)
            return np.stack(channels, axis=0)

        elif return_format == 'separate':
            # Return as separate matrices
            result = {
                'frequency': freq_matrix,
                'mean': mean_matrix,
                'std': std_matrix,
                'skewness': skew_matrix,
                'kurtosis': kurt_matrix
            }
            if include_mask:
                result['mask'] = mask_matrix
            return result

        elif return_format == 'tabular':
            # Return as tabular format for traditional ML models
            return self._to_tabular_format(kmer_with_positions)

    def _to_tabular_format(self, kmer_with_positions: Dict[str, List[int]]) -> Dict[str, float]:
        features = {}

        # Generate all possible k-mers
        all_kmers = list(self._kmer_positions.keys())

        for kmer in all_kmers:
            if kmer in kmer_with_positions:
                positions = kmer_with_positions[kmer]
                freq = len(positions)
                mean, std_dev, skewness, kurtosis = self._calculate_statistics(positions)
            else:
                freq, mean, std_dev, skewness, kurtosis = 0, 0, 0, 0, 0

            # Create column names
            features[f"{kmer}_frequency"] = freq
            features[f"{kmer}_mean"] = mean
            features[f"{kmer}_std"] = std_dev
            features[f"{kmer}_skewness"] = skewness
            features[f"{kmer}_kurtosis"] = kurtosis

        return features

    def get_feature_names(self) -> List[str]:
        feature_names = []
        suffixes = ['frequency', 'mean', 'std', 'skewness', 'kurtosis']

        for kmer in sorted(self._kmer_positions.keys()):
            for suffix in suffixes:
                feature_names.append(f"{kmer}_{suffix}")

        return feature_names


def read_fasta(file_path: str) -> Dict[str, str]:
    sequences = {}
    current_id = None
    current_seq = []

    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_id:
                        sequences[current_id] = ''.join(current_seq)
                    current_id = line[1:]
                    current_seq = []
                else:
                    current_seq.append(line)

            if current_id:
                sequences[current_id] = ''.join(current_seq)

    except FileNotFoundError:
        raise FileNotFoundError(f"FASTA file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading FASTA file: {e}")

    return sequences
