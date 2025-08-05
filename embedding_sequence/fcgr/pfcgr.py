from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from .fcgr import FCGR


class PFCGR(FCGR):
    """Positional Frequency CGR

    Extends FCGR by incorporating positional information through
    statistical measures of k-mer positions:
    - Mean position
    - Standard deviation of positions
    - Skewness of positions
    - Kurtosis of positions

    Returns 5 matrices (2^k x 2^k each):
    - Frequency matrix (from FCGR)
    - Mean position matrix
    - Standard deviation matrix
    - Skewness matrix
    - Kurtosis matrix
    """

    def __init__(self, k: int):
        super().__init__(k)
        # Dictionary to store positions of each k-mer
        self.kmer_positions = defaultdict(list)

    def __call__(self, sequence: str):
        """
        Given a DNA sequence, returns 5 matrices with frequency and
        positional statistics for each k-mer.

        Returns:
            dict: Dictionary containing 5 matrices:
                - 'frequency': k-mer frequency matrix
                - 'mean': mean position matrix
                - 'std': standard deviation matrix
                - 'skewness': skewness matrix
                - 'kurtosis': kurtosis matrix
        """
        # Reset k-mer positions
        self.kmer_positions = defaultdict(list)

        # Count k-mers and record their positions
        self.count_kmers_with_positions(sequence)

        # Create empty arrays for all statistics
        array_size = int(2 ** self.k)
        freq_matrix = np.zeros((array_size, array_size))
        mean_matrix = np.zeros((array_size, array_size))
        std_matrix = np.zeros((array_size, array_size))
        skew_matrix = np.zeros((array_size, array_size))
        kurt_matrix = np.zeros((array_size, array_size))

        # Calculate statistics for each k-mer
        for kmer, positions in self.kmer_positions.items():
            if "N" in kmer:  # Skip k-mers with N
                continue

            # Get pixel position for this k-mer
            pos_x, pos_y = self.kmer2pixel[kmer]
            idx_x, idx_y = int(pos_x) - 1, int(pos_y) - 1

            # Frequency
            freq_matrix[idx_x, idx_y] = len(positions)

            # Positional statistics
            if len(positions) >= 1:
                positions_array = np.array(positions)

                # Mean
                mean_matrix[idx_x, idx_y] = np.mean(positions_array)

                # Standard deviation (requires at least 2 values)
                if len(positions) >= 2:
                    std_matrix[idx_x, idx_y] = np.std(positions_array, ddof=1)

                # Skewness and Kurtosis (requires at least 3 values)
                if len(positions) >= 3:
                    skew_matrix[idx_x, idx_y] = stats.skew(positions_array)

                # Kurtosis (requires at least 4 values)
                if len(positions) >= 4:
                    kurt_matrix[idx_x, idx_y] = stats.kurtosis(positions_array)

        # Normalize matrices (min-max normalization as mentioned in paper)
        freq_matrix = self._normalize_matrix(freq_matrix)
        mean_matrix = self._normalize_matrix(mean_matrix)
        std_matrix = self._normalize_matrix(std_matrix)
        skew_matrix = self._normalize_matrix(skew_matrix)
        kurt_matrix = self._normalize_matrix(kurt_matrix)

        return {
            'frequency': freq_matrix,
            'mean': mean_matrix,
            'std': std_matrix,
            'skewness': skew_matrix,
            'kurtosis': kurt_matrix
        }

    def count_kmers_with_positions(self, sequence: str):
        """Count k-mers and record their positions in the sequence"""
        last_j = len(sequence) - self.k + 1

        for i in range(last_j):
            kmer = sequence[i:(i + self.k)]
            if "N" not in kmer:
                # Store the starting position (0-indexed)
                self.kmer_positions[kmer].append(i)

        # Also update freq_kmer for compatibility
        self.freq_kmer = defaultdict(int)
        for kmer, positions in self.kmer_positions.items():
            self.freq_kmer[kmer] = len(positions)

    def _normalize_matrix(self, matrix, method='min-max'):
        if method == 'min-max':
            min_val = np.min(matrix)
            max_val = np.max(matrix)

            if max_val - min_val > 0:
                return (matrix - min_val) / (max_val - min_val)
            else:
                return matrix

        elif method == 'z-score':
            mean_val = np.mean(matrix)
            std_val = np.std(matrix)

            if std_val > 0:
                return (matrix - mean_val) / std_val
            else:
                # Nếu std = 0 (tất cả giá trị giống nhau), trả về ma trận zero
                return np.zeros_like(matrix)

        else:
            raise ValueError(f"Unsupported normalization method: {method}. Supported methods: 'min-max', 'z-score'")

    def get_tabular_features(self, sequence: str):
        """
        Get features in tabular format for ML models like LR and RF.

        Returns:
            dict: Dictionary with k-mer as key and dict of statistics as value
        """
        # Get matrices
        matrices = self.__call__(sequence)

        # Convert to tabular format
        features = {}

        for kmer in self.kmers:
            if "N" in kmer:
                continue

            pos_x, pos_y = self.kmer2pixel[kmer]
            idx_x, idx_y = int(pos_x) - 1, int(pos_y) - 1

            # Only include features for k-mers that exist in the sequence
            if kmer in self.kmer_positions and len(self.kmer_positions[kmer]) > 0:
                features[f"{kmer}_frequency"] = matrices['frequency'][idx_x, idx_y]
                features[f"{kmer}_mean"] = matrices['mean'][idx_x, idx_y]

                if len(self.kmer_positions[kmer]) >= 2:
                    features[f"{kmer}_std"] = matrices['std'][idx_x, idx_y]

                if len(self.kmer_positions[kmer]) >= 3:
                    features[f"{kmer}_skewness"] = matrices['skewness'][idx_x, idx_y]

                if len(self.kmer_positions[kmer]) >= 4:
                    features[f"{kmer}_kurtosis"] = matrices['kurtosis'][idx_x, idx_y]

        return features

    def get_multichannel_image(self, sequence: str):
        """
        Get PFCGR as a multi-channel image for CNN models.

        Returns:
            np.ndarray: Array of shape (5, 2^k, 2^k) with 5 channels
        """
        matrices = self.__call__(sequence)

        # Stack matrices as channels
        multichannel = np.stack([
            matrices['frequency'],
            matrices['mean'],
            matrices['std'],
            matrices['skewness'],
            matrices['kurtosis']
        ], axis=0)

        return multichannel

    def visualize_pfcgr(self, sequence: str, title: str = "PFCGR Representation", figsize=(20, 4)):
        """
        Visualize all 5 channels of PFCGR representation of a sequence

        Args:
            sequence: DNA sequence to visualize
            title: Title for the plot
            figsize: Figure size (width, height)
        """
        seq = self.preprocessing(sequence)
        matrices = self.__call__(seq)

        # Channel names and colormaps
        channels = ['frequency', 'mean', 'std', 'skewness', 'kurtosis']
        channel_titles = ['Frequency', 'Mean Position', 'Standard Deviation', 'Skewness', 'Kurtosis']
        colormaps = ['hot', 'viridis', 'plasma', 'coolwarm', 'seismic']

        # Create subplots
        fig, axes = plt.subplots(1, 5, figsize=figsize)
        fig.suptitle(f"{title}\nSequence length: {len(sequence)}, k={self.k}", fontsize=14)

        for i, (channel, channel_title, cmap) in enumerate(zip(channels, channel_titles, colormaps)):
            matrix = matrices[channel]

            # Plot the matrix
            im = axes[i].imshow(matrix, cmap=cmap, interpolation='nearest', aspect='equal')
            axes[i].set_title(channel_title, fontsize=12)
            axes[i].set_xlabel('X coordinate')
            if i == 0:  # Only show y-label for first subplot
                axes[i].set_ylabel('Y coordinate')

            # Add colorbar
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

            # Add statistics text
            non_zero_count = np.count_nonzero(matrix)
            matrix_stats = f'Non-zero: {non_zero_count}\nMax: {matrix.max():.3f}\nMin: {matrix.min():.3f}'
            axes[i].text(0.02, 0.98, matrix_stats, transform=axes[i].transAxes,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                         fontsize=8)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def preprocessing(seq):
        seq = seq.upper()
        for letter in "BDEFHIJKLMOPQRSUVWXYZ":
            seq = seq.replace(letter, "N")
        return seq


# Example usage function
def example_usage():
    """Example of how to use PFCGR"""
    # Create PFCGR object for 4-mers
    pfcgr = PFCGR(k=4)

    # Example DNA sequence
    sequence = "ATCGATCGATCGATCGATCGATCGATCGATCG"

    # Get all matrices
    matrices = pfcgr(sequence)

    print("Frequency matrix shape:", matrices['frequency'].shape)
    print("Mean matrix shape:", matrices['mean'].shape)

    # Get tabular features for ML
    features = pfcgr.get_tabular_features(sequence)
    print("\nNumber of features:", len(features))
    print("Sample features:", list(features.keys())[:5])

    # Get multi-channel image for CNN
    multichannel = pfcgr.get_multichannel_image(sequence)
    print("\nMulti-channel image shape:", multichannel.shape)

    return pfcgr, matrices, features, multichannel
