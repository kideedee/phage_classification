import pickle
from collections import Counter
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from joblib import Parallel
from sklearn.model_selection import train_test_split

from common.env_config import config
from embedding_sequence.encoding_sequence import DNASequenceProcessor
from embedding_sequence.fcgr.fcgr_processor import FCGREncoder, load_and_preprocess_data
from logger.phg_cls_log import log


class PhageFCGRProcessor:
    """
    Processor chuyên dụng cho dữ liệu phage sử dụng FCGR
    """

    def __init__(self, k=8, resolution=64):
        self.encoder = FCGREncoder(k=k, resolution=resolution)
        self.k = k
        self.resolution = resolution

    def analyze_sequence_lengths(self, sequences: List[str]) -> dict:
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

    def visualize_data_distribution(self, sequences: List[str], targets: List[int]):
        """
        Visualize phân phối dữ liệu
        """
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

    def process_dataset(self, sequences, targets, save_embeddings: bool = True) -> dict:
        """
        Xử lý toàn bộ dataset và tạo FCGR embeddings
        """

        print("Analyzing data distribution...")
        self.visualize_data_distribution(sequences, targets)

        print("Creating FCGR embeddings...")
        embeddings = self.encoder.encode_sequences(sequences)

        # Reshape embeddings thành vector để có thể sử dụng với ML models
        embeddings_flat = embeddings.reshape(embeddings.shape[0], -1)

        print(f"FCGR embeddings shape: {embeddings.shape}")
        print(f"Flattened embeddings shape: {embeddings_flat.shape}")

        # Tạo train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings_flat, targets, test_size=0.2, random_state=42, stratify=targets
        )

        results = {
            'sequences': sequences,
            'targets': targets,
            'embeddings_2d': embeddings,
            'embeddings_flat': embeddings_flat,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

        if save_embeddings:
            print("Saving embeddings...")
            np.save(f'fcgr_embeddings_k{self.k}_res{self.resolution}.npy', embeddings)
            np.save(f'fcgr_embeddings_flat_k{self.k}_res{self.resolution}.npy', embeddings_flat)

            with open(f'fcgr_dataset_k{self.k}_res{self.resolution}.pkl', 'wb') as f:
                pickle.dump(results, f)

        return results

    def visualize_sample_fcgrs(self, sequences: List[str], targets: List[int], n_samples: int = 6):
        """
        Visualize một số FCGR samples
        """
        # Chọn random samples từ mỗi class
        temperate_idx = [i for i, t in enumerate(targets) if t == 0]
        virulent_idx = [i for i, t in enumerate(targets) if t == 1]

        selected_idx = (np.random.choice(temperate_idx, n_samples // 2, replace=False).tolist() +
                        np.random.choice(virulent_idx, n_samples // 2, replace=False).tolist())

        fig, axes = plt.subplots(2, n_samples // 2, figsize=(15, 8))
        axes = axes.flatten()

        for i, idx in enumerate(selected_idx):
            sequence = sequences[idx]
            target = targets[idx]
            fcgr_matrix = self.encoder._sequence_to_fcgr(sequence)

            axes[i].imshow(fcgr_matrix, cmap='hot', interpolation='nearest')
            axes[i].set_title(f"{'Temperate' if target == 0 else 'Virulent'}\nLen: {len(sequence)}bp")
            axes[i].axis('off')

        plt.suptitle(f'Sample FCGR Representations (k={self.k}, resolution={self.resolution})')
        plt.tight_layout()
        plt.show()


def experiment_with_parameters(sequences, targets, k_values: List[int] = [6, 8, 10],
                               resolutions: List[int] = [64, 128]):
    """
    Thử nghiệm với các tham số k và resolution khác nhau
    """
    print("=== FCGR PARAMETER EXPERIMENTS ===")

    results = {}

    for k in k_values:
        for resolution in resolutions:
            print(f"\nTesting k={k}, resolution={resolution}")

            processor = PhageFCGRProcessor(k=k, resolution=resolution)

            # Tạo embeddings cho subset nhỏ để test
            test_sequences = sequences[:100]  # Test với 100 sequences đầu
            embeddings = processor.encoder.encode_sequences(test_sequences)

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


# Main processing script
if __name__ == "__main__":
    train_file = config.TRAIN_DATA_FOLD_1_CSV_FILE
    valid_file = config.TEST_DATA_FOLD_1_CSV_FILE

    dna_bert_2_processor = DNASequenceProcessor(
        min_size=1200,
        max_size=1800,
        overlap_percent=30,
    )

    train_df, val_df = dna_bert_2_processor.load_and_clean_data(train_file, valid_file)
    X_train, y_train, X_val, y_val = dna_bert_2_processor.window_and_extract_features(train_df, val_df)
    X_train_aug, y_train_aug = dna_bert_2_processor.reverse_complement_augmentation(X_train, y_train)
    log.info(f"Original training set size: {len(X_train_aug)}")

    X_train_aug_df = pd.DataFrame({'sequence': X_train_aug})
    index_array = np.array(X_train_aug_df.index).reshape(-1, 1)  # Convert indexes to 2D array
    undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    index_resampled, y_resampled = undersampler.fit_resample(index_array, y_train_aug)
    X_resampled = X_train_aug_df.iloc[index_resampled.flatten()]['sequence'].values
    log.info(f"Resampled training set size: {len(X_resampled)}")

    processed_train_df = pd.DataFrame(zip(X_resampled, y_resampled), columns=["sequence", "target"])
    processed_val_df = pd.DataFrame(zip(X_val, y_val), columns=["sequence", "target"])

    # Thử nghiệm tham số tối ưu
    print("Testing different parameters...")
    param_results = experiment_with_parameters(X_resampled, y_resampled)

    # Xử lý với tham số được chọn
    print("\nProcessing with optimal parameters...")
    processor = PhageFCGRProcessor(k=8, resolution=64)
    results = processor.process_dataset(X_resampled, y_resampled)

    # Visualize một số samples
    processor.visualize_sample_fcgrs(
        results['sequences'],
        results['targets'],
        n_samples=6
    )

    print("\nFCGR processing completed!")
    print(f"Ready for machine learning with {results['embeddings_flat'].shape[1]} features per sequence")
