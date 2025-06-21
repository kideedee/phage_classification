from typing import List, Tuple, Any, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from numpy import ndarray, dtype, floating, complexfloating
from numpy._typing import _64Bit

from embedding_sequence.abstract_embedding import AbstractEmbedding


class FCGREmbedding(AbstractEmbedding):

    def __init__(self, embedding_type, min_size, max_size, overlap_percent, kmer, resolution, fold, is_train):
        super().__init__(embedding_type=embedding_type, min_size=min_size, max_size=max_size,
                         overlap_percent=overlap_percent, fold=fold, is_train=is_train)
        self.kmer = kmer
        self.resolution = resolution
        self.nucleotide_map = {
            'A': (0, 0),  # Bottom-left
            'T': (0, 1),  # Top-left
            'G': (1, 0),  # Bottom-right
            'C': (1, 1)  # Top-right
        }

    def encode_sequences(self, sequences: List[str], labels: List[str]) -> tuple[
        ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]
    ]:

        fcgr_representations = []
        result_labels = []

        # for i, sequence in enumerate(sequences):
        #     if i % 100 == 0:
        #         print(f"Processing sequence {i + 1}/{len(sequences)}")
        #
        #     fcgr_matrix = self._sequence_to_fcgr(sequence)
        #     fcgr_representations.append(fcgr_matrix)
        #     result_labels.append(labels[i])

        df = pd.DataFrame(zip(sequences, labels), columns=['sequence', 'label'])
        results = Parallel(n_jobs=10)(
            delayed(self._sequence_to_fcgr)(
                (idx, row)
            ) for idx, row in df.iterrows()
        )

        for result in results:
            if result:
                fcgr_representations.append(result[0])
                result_labels.append(result[1])

        return np.array(fcgr_representations), np.array(result_labels)

    def _sequence_to_fcgr(self, row_tuple) -> Union[ndarray[Any, dtype[floating[_64Bit]]], tuple[Union[Union[
        ndarray[Any, dtype[floating[Any]]], ndarray[Any, dtype[complexfloating[Any, Any]]], ndarray[
            Any, dtype[floating[_64Bit]]]], Any], Any]]:
        """
        Chuyển đổi một chuỗi DNA thành ma trận FCGR
        """

        idx, row = row_tuple
        sequence = row['sequence']
        label = row['label']

        # Loại bỏ các ký tự không phải nucleotide
        clean_sequence = ''.join([c for c in sequence.upper() if c in 'ATGC'])

        if len(clean_sequence) < self.kmer:
            # Nếu sequence quá ngắn, trả về ma trận zeros
            return np.zeros((self.resolution, self.resolution))

        # Khởi tạo ma trận frequency
        fcgr_matrix = np.zeros((self.resolution, self.resolution))

        # Tạo tất cả k-mers từ sequence
        kmers = []
        for i in range(len(clean_sequence) - self.kmer + 1):
            kmer = clean_sequence[i:i + self.kmer]
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

        return fcgr_matrix, label

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

    def visualize_data_distribution(self):
        """
        Visualize phân phối dữ liệu
        """

        sequences = self.x_resampled
        targets = self.y_resampled

        stats, lengths = self.analyze_sequence_lengths(sequences)

        # Plot 1: Phân phối độ dài sequences
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.hist(lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Sequence Length (bp)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Sequence Lengths')
        plt.grid(True, alpha=0.3)

        # Thêm vertical lines cho các nhóm
        plt.axvline(400, color='red', linestyle='--', alpha=0.7, label='Group boundaries')
        plt.axvline(800, color='red', linestyle='--', alpha=0.7)
        plt.axvline(1200, color='red', linestyle='--', alpha=0.7)
        plt.legend()

        # Plot 2: Phân phối targets
        plt.subplot(1, 3, 2)
        target_counts = pd.Series(targets).value_counts()
        plt.bar(['Temperate (0)', 'Virulent (1)'], target_counts.values,
                color=['lightcoral', 'lightgreen'], alpha=0.7)
        plt.ylabel('Count')
        plt.title('Target Distribution')
        plt.grid(True, alpha=0.3)

        # Plot 3: Phân phối targets theo nhóm độ dài
        plt.subplot(1, 3, 3)
        df = pd.DataFrame({'length': lengths, 'target': targets})
        df['group'] = pd.cut(df['length'], bins=[0, 400, 800, 1200, 1800, float('inf')],
                             labels=['A', 'B', 'C', 'D', 'E'])

        group_target = df.groupby(['group', 'target']).size().unstack(fill_value=0)
        group_target.plot(kind='bar', ax=plt.gca(), color=['lightcoral', 'lightgreen'])
        plt.ylabel('Count')
        plt.title('Target Distribution by Length Groups')
        plt.legend(['Temperate', 'Virulent'])
        plt.xticks(rotation=0)

        plt.tight_layout()
        plt.show()

        # In thống kê chi tiết
        print("=== SEQUENCE LENGTH STATISTICS ===")
        for group, stat in stats.items():
            print(f"{group}: {stat['count']} sequences, "
                  f"mean={stat['mean']:.1f}±{stat['std']:.1f}bp")

    @staticmethod
    def analyze_sequence_lengths(sequences) -> dict:
        """
        Phân tích độ dài sequences theo các nhóm A, B, C, D
        """
        lengths = [len(seq) for seq in sequences]

        groups = {
            'A (100-400bp)': [l for l in lengths if 100 <= l <= 400],
            'B (400-800bp)': [l for l in lengths if 400 < l <= 800],
            'C (800-1200bp)': [l for l in lengths if 800 < l <= 1200],
            'D (1200-1800bp)': [l for l in lengths if 1200 < l <= 1800]
        }

        stats = {}
        for group, group_lengths in groups.items():
            if group_lengths:
                stats[group] = {
                    'count': len(group_lengths),
                    'mean': np.mean(group_lengths),
                    'std': np.std(group_lengths),
                    'min': min(group_lengths),
                    'max': max(group_lengths)
                }

        return stats, lengths

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

    # @staticmethod
    def experiment_with_parameters(self, sequences, targets, k_values: List[int] = [6, 8, 10],
                                   resolutions: List[int] = [64, 128]):
        """
        Thử nghiệm với các tham số k và resolution khác nhau
        """
        print("=== FCGR PARAMETER EXPERIMENTS ===")

        results = {}

        for k in k_values:
            for resolution in resolutions:
                print(f"\nTesting k={k}, resolution={resolution}")

                # processor = PhageFCGRProcessor(k=k, resolution=resolution)

                # Tạo embeddings cho subset nhỏ để test
                test_sequences = sequences[:100]  # Test với 100 sequences đầu
                embeddings = self.encode_sequences(test_sequences)

                # Tính một số metrics cơ bản
                embedding_stats = {
                    'shape': embeddings.shape,
                    'non_zero_ratio': np.mean(embeddings > 0),
                    'mean_intensity': np.mean(embeddings),
                    'std_intensity': np.std(embeddings)
                }

                results[f'k{k}_res{resolution}'] = embedding_stats
                print(f"Non-zero ratio: {embedding_stats['non_zero_ratio']:.3f}")
                print(f"Mean intensity: {embedding_stats['mean_intensity']:.6f}")

        return results
