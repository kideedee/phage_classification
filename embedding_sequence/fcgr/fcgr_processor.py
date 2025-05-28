import warnings
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class FCGREncoder:
    """
    Frequency Chaos Game Representation encoder cho chuỗi DNA
    """

    def __init__(self, k=8, resolution=64):
        """
        Args:
            k: độ dài k-mer (thường từ 6-10)
            resolution: độ phân giải của ma trận FCGR (64x64, 128x128, ...)
        """
        self.k = k
        self.resolution = resolution

        # Mapping nucleotides to coordinates
        self.nucleotide_map = {
            'A': (0, 0),  # Bottom-left
            'T': (0, 1),  # Top-left
            'G': (1, 0),  # Bottom-right
            'C': (1, 1)  # Top-right
        }

    def _get_cgr_coordinates(self, sequence: str) -> List[Tuple[float, float]]:
        """
        Tính toán tọa độ CGR cho một chuỗi DNA
        """
        # Khởi tạo vị trí ban đầu ở giữa
        x, y = 0.5, 0.5
        coordinates = [(x, y)]

        for nucleotide in sequence.upper():
            if nucleotide in self.nucleotide_map:
                corner_x, corner_y = self.nucleotide_map[nucleotide]
                # Di chuyển đến giữa vị trí hiện tại và góc của nucleotide
                x = (x + corner_x) / 2
                y = (y + corner_y) / 2
                coordinates.append((x, y))

        return coordinates

    def _sequence_to_fcgr(self, sequence: str) -> np.ndarray:
        """
        Chuyển đổi một chuỗi DNA thành ma trận FCGR
        """
        # Loại bỏ các ký tự không phải nucleotide
        clean_sequence = ''.join([c for c in sequence.upper() if c in 'ATGC'])

        if len(clean_sequence) < self.k:
            # Nếu sequence quá ngắn, trả về ma trận zeros
            return np.zeros((self.resolution, self.resolution))

        # Khởi tạo ma trận frequency
        fcgr_matrix = np.zeros((self.resolution, self.resolution))

        # Tạo tất cả k-mers từ sequence
        kmers = []
        for i in range(len(clean_sequence) - self.k + 1):
            kmer = clean_sequence[i:i + self.k]
            if all(c in 'ATGC' for c in kmer):
                kmers.append(kmer)

        if not kmers:
            return fcgr_matrix

        # Tính tọa độ CGR cho mỗi k-mer và cập nhật frequency matrix
        for kmer in kmers:
            coordinates = self._get_cgr_coordinates(kmer)
            if coordinates:
                # Lấy tọa độ cuối cùng
                final_x, final_y = coordinates[-1]

                # Chuyển đổi tọa độ CGR thành indices của ma trận
                i = int(final_y * self.resolution)
                j = int(final_x * self.resolution)

                # Đảm bảo indices trong phạm vi hợp lệ
                i = min(i, self.resolution - 1)
                j = min(j, self.resolution - 1)

                fcgr_matrix[i, j] += 1

        # Normalize frequency matrix
        if fcgr_matrix.sum() > 0:
            fcgr_matrix = fcgr_matrix / fcgr_matrix.sum()

        return fcgr_matrix

    def encode_sequences(self, sequences: List[str]) -> np.ndarray:
        """
        Encode một list các sequences thành FCGR representations

        Returns:
            numpy array có shape (n_sequences, resolution, resolution)
        """
        fcgr_representations = []

        for i, sequence in enumerate(sequences):
            if i % 100 == 0:
                print(f"Processing sequence {i + 1}/{len(sequences)}")

            fcgr_matrix = self._sequence_to_fcgr(sequence)
            fcgr_representations.append(fcgr_matrix)

        return np.array(fcgr_representations)

    def visualize_fcgr(self, sequence: str, title: str = "FCGR Representation"):
        """
        Visualize FCGR representation của một sequence
        """
        fcgr_matrix = self._sequence_to_fcgr(sequence)

        plt.figure(figsize=(8, 8))
        plt.imshow(fcgr_matrix, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Frequency')
        plt.title(f"{title}\nSequence length: {len(sequence)}")
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.show()

        return fcgr_matrix


# Hàm tiện ích để load và preprocess dữ liệu
def load_and_preprocess_data(csv_file: str) -> Tuple[List[str], List[int]]:
    """
    Load dữ liệu từ CSV file

    Args:
        csv_file: đường dẫn đến file CSV chứa columns 'sequence' và 'target'

    Returns:
        sequences: list các DNA sequences
        targets: list các labels (0: temperate, 1: virulent)
    """
    df = pd.read_csv(csv_file)
    sequences = df['sequence'].tolist()
    targets = df['target'].tolist()

    print(f"Loaded {len(sequences)} sequences")
    print(f"Target distribution: {pd.Series(targets).value_counts().to_dict()}")

    return sequences, targets


def create_fcgr_embeddings(sequences: List[str], k: int = 8, resolution: int = 64) -> np.ndarray:
    """
    Tạo FCGR embeddings cho tất cả sequences

    Args:
        sequences: list các DNA sequences
        k: độ dài k-mer
        resolution: độ phân giải FCGR matrix

    Returns:
        FCGR embeddings với shape (n_sequences, resolution, resolution)
    """
    encoder = FCGREncoder(k=k, resolution=resolution)
    embeddings = encoder.encode_sequences(sequences)

    print(f"Created FCGR embeddings with shape: {embeddings.shape}")

    return embeddings


# Example usage
if __name__ == "__main__":
    # Ví dụ sử dụng
    sample_sequences = [
        "ATGCGATCGATCGATCGATCGATCGATCG",
        "AAAAAATTTTTCCCCCGGGGG",
        "ATCGATCGATCGATCGATCGATCGATCGATCG"
    ]

    # Tạo encoder
    encoder = FCGREncoder(k=6, resolution=64)

    # Visualize một vài examples
    for i, seq in enumerate(sample_sequences):
        encoder.visualize_fcgr(seq, f"Sample sequence {i + 1}")

    # Encode tất cả sequences
    embeddings = encoder.encode_sequences(sample_sequences)
    print(f"Embeddings shape: {embeddings.shape}")
